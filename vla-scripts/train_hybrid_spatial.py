"""Step-based LIBERO Hybrid training on one or four CUDA workers."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import runpy
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    # Load the stateless helper without executing prismatic's eager package imports
    # before NCCL initialization. CLI and metadata share this one registry.
    config = runpy.run_path(str(Path(__file__).resolve().parents[1] / "prismatic/training/hybrid_formal.py"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", default=config["DEFAULT_HYBRID_DATASET_KEY"],
                        choices=config["HYBRID_DATASET_CONFIGS"])
    for name in ("vlm-path", "hf-config", "data-root", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--flow-head-learning-rate", type=float, default=None)
    parser.add_argument("--lr-decay-step", type=int, default=30000)
    parser.add_argument("--lr-decay-factor", type=float, default=.1)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--benchmark-warmup-steps", type=int,
                        help="Opt-in CUDA timing; exclude this many initial updates from throughput")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-max-steps-extension", action="store_true")
    parser.add_argument("--no-image-aug", dest="image_aug", action="store_false")
    parser.set_defaults(image_aug=True)
    args = parser.parse_args(argv)
    if args.allow_max_steps_extension and args.resume is None:
        parser.error("--allow-max-steps-extension requires --resume")
    try:
        args.flow_head_learning_rate = config["effective_flow_head_learning_rate"](
            args.learning_rate, args.flow_head_learning_rate)
    except ValueError as error:
        parser.error(str(error))
    if min(args.max_steps, args.save_every, args.log_every, args.shuffle_buffer_size,
           args.lr_decay_step, args.per_device_batch_size) < 1 or args.seed < 0:
        parser.error("Step/buffer sizes must be positive and seed nonnegative")
    if args.benchmark_warmup_steps is not None and not 0 <= args.benchmark_warmup_steps < args.max_steps:
        parser.error("benchmark-warmup-steps must be nonnegative and smaller than max-steps")
    return args


def rank_zero_io(operation):
    """Propagate writer failures at manifest/log/checkpoint publication points."""
    status = [None]
    if dist.get_rank() == 0:
        try:
            operation()
            status[0] = {"error": None}
        except Exception as error:
            status[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(status, src=0)
    if status[0]["error"]:
        raise RuntimeError(f"Rank-zero output failed: {status[0]['error']}")


def make_data(args, tokenizer, images, rank, world_size):
    from torch.utils.data import DataLoader
    from prismatic.vla.datasets.hybrid_datasets import HybridRLDSDataset, HybridRLDSBatchTransform, PaddedCollatorForHybridFlow
    from prismatic.vla.hybrid_normalization import HybridZScoreNormalizer

    dataset = HybridRLDSDataset(args.data_root, args.dataset_key,
        HybridRLDSBatchTransform(tokenizer, images.apply_transform), resize_resolution=(224, 224),
        shuffle_buffer_size=args.shuffle_buffer_size, train=True, image_aug=args.image_aug,
        rank=rank, world_size=world_size)
    if args.dataset_key not in dataset.dataset_statistics:
        raise ValueError(f"Dataset statistics do not contain requested dataset key: {args.dataset_key}")
    statistics = dataset.dataset_statistics[args.dataset_key]
    loader = DataLoader(dataset, batch_size=args.per_device_batch_size, num_workers=0,
                        collate_fn=PaddedCollatorForHybridFlow(tokenizer.pad_token_id))
    return loader, HybridZScoreNormalizer(statistics), statistics, str(dataset.resolved_source_split)


def training_loop(ddp_model, loader, normalizer, optimizer, *, args, global_step, log_callback, checkpoint_callback):
    from prismatic.training.hybrid_formal import (
        hybrid_learning_rate, set_optimizer_learning_rates, effective_flow_head_learning_rate, checkpoint_due,
    )
    from prismatic.training.hybrid_multistep import hybrid_training_step

    warmup = getattr(args, "benchmark_warmup_steps", None)
    if warmup is not None and warmup >= args.max_steps - global_step:
        raise ValueError("Benchmark requires updates after warmup in this process segment")
    iterator = iter(loader)
    timings = []
    peak_allocated = peak_reserved = 0
    if warmup is not None:
        torch.cuda.reset_peak_memory_stats()
    while global_step < args.max_steps:
        if warmup is not None:
            torch.cuda.synchronize()
            started = time.perf_counter()
        batch = next(iterator)
        if warmup is not None:
            data_wait_sec = time.perf_counter() - started
        B = args.per_device_batch_size
        if batch["actions"].shape != (B, 10, 7) or batch["proprio"].shape != (B, 8):
            raise ValueError(f"Training requires {B} H10/A7/P8 samples per rank")
        lr = hybrid_learning_rate(global_step, base_lr=args.learning_rate,
                                  decay_step=args.lr_decay_step, decay_factor=args.lr_decay_factor)
        head_lr = hybrid_learning_rate(global_step,
            base_lr=effective_flow_head_learning_rate(args.learning_rate, getattr(args, "flow_head_learning_rate", None)),
            decay_step=args.lr_decay_step, decay_factor=args.lr_decay_factor)
        set_optimizer_learning_rates(optimizer, vlm_lr=lr, flow_head_lr=head_lr)
        if warmup is not None:
            compute_started = time.perf_counter()
        diagnostics = hybrid_training_step(ddp_model, batch, normalizer, optimizer, device_type="cuda",
                                           autocast_dtype=torch.bfloat16, autocast_enabled=True)
        if warmup is not None:
            torch.cuda.synchronize()
            compute_sec = time.perf_counter() - compute_started
        global_step += 1
        optimizer.zero_grad(set_to_none=True)
        del batch
        if warmup is not None:
            timings.append((time.perf_counter() - started, data_wait_sec, compute_sec))
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
        if (global_step % args.log_every == 0 or global_step == args.max_steps
                or global_step in (args.lr_decay_step, args.lr_decay_step + 1)):
            diagnostics.update(vlm_learning_rate=lr, flow_head_learning_rate=head_lr)
            log_callback(global_step, lr, diagnostics)
        del diagnostics
        if checkpoint_due(global_step, max_steps=args.max_steps, save_every=args.save_every):
            checkpoint_callback(global_step)
    if warmup is not None:
        measured = timings[warmup:]
        seconds, data_wait, compute = (sum(values) / len(measured) for values in zip(*measured))
        print(json.dumps(dict(benchmark_rank=dist.get_rank(), measured_steps=len(measured),
            mean_data_wait_sec=data_wait, mean_compute_sec=compute,
            data_wait_fraction=data_wait / seconds, compute_fraction=compute / seconds,
            seconds_per_optimizer_step=seconds, samples_per_sec=args.per_device_batch_size / seconds,
            cuda_peak_allocated_gib=peak_allocated / 2**30,
            cuda_peak_reserved_gib=peak_reserved / 2**30)), flush=True)
    return global_step


def write_log(run_dir, step, lr, diagnostics, device):
    keys = ("loss", "t_mean", "velocity_rms", "encoder_grad_norm", "flow_head_grad_norm")
    values = torch.tensor([diagnostics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    torch.cuda.synchronize(device)
    memory = torch.tensor([torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(device)],
                          device=device, dtype=torch.float64) / 2**30
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    names = ("loss_mean", "t_mean", "velocity_rms_mean", "encoder_grad_norm_mean", "flow_head_grad_norm_mean")
    record = dict(zip(names, values.cpu().tolist()))
    record.update(global_step=step, learning_rate=lr, vlm_learning_rate=lr,
                  flow_head_learning_rate=diagnostics["flow_head_learning_rate"], cuda_allocated_gib=memory[0].item(),
                  cuda_peak_allocated_gib=memory[1].item(), timestamp=datetime.now(timezone.utc).isoformat())
    def append():
        with (run_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    rank_zero_io(append)


def main():
    args = parse_args()
    if not all(path.is_dir() for path in (args.vlm_path, args.hf_config, args.data_root)):
        raise ValueError("Native VLM, HF config and RLDS data directories must exist")
    if not (args.data_root / args.dataset_key).is_dir():
        raise FileNotFoundError(f"Requested RLDS dataset {args.dataset_key} is missing: {args.data_root / args.dataset_key}")
    if int(os.environ.get("WORLD_SIZE", 0)) not in (1, 4) or not torch.cuda.is_available():
        raise RuntimeError("Use torchrun with one or four CUDA workers")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=15))
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        # Prismatic imports can initialize distributed state; NCCL must already be configured.
        from prismatic.training.hybrid_formal import hybrid_learning_rate, build_formal_metadata, prepare_run_manifest, checkpoint_name
        from prismatic.training.hybrid_formal import set_group_learning_rates
        hybrid_learning_rate(0, base_lr=args.learning_rate, decay_step=args.lr_decay_step, decay_factor=args.lr_decay_factor)
        from hybrid_ddp_smoke import load_assets, gather_report, require_all
        from hybrid_checkpoint_smoke import state_checks
        from hybrid_multistep_smoke import optimizer_step_range
        from prismatic.training.hybrid_ddp import HybridFlowTrainingModule
        from prismatic.training.hybrid_step import hybrid_parameter_groups
        from prismatic.training.hybrid_checkpoint import save_hybrid_checkpoint, load_hybrid_checkpoint

        rank, world_size = dist.get_rank(), dist.get_world_size()
        if rank == 0:
            print(json.dumps(dict(world_size=world_size, per_device_batch_size=args.per_device_batch_size,
                vlm_learning_rate=args.learning_rate, flow_head_learning_rate=args.flow_head_learning_rate,
                gradient_accumulation_steps=1,
                effective_global_batch_size=world_size * args.per_device_batch_size)), flush=True)
        torch.manual_seed(args.seed)
        encoder, head, tokenizer, images = load_assets(args.vlm_path, args.hf_config)
        model = HybridFlowTrainingModule(encoder, head).to(device)
        groups = hybrid_parameter_groups(model.encoder, model.flow_head)
        set_group_learning_rates(groups, vlm_lr=args.learning_rate, flow_head_lr=args.flow_head_learning_rate)
        policy = {name: p.requires_grad for name, p in model.named_parameters()}
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
        loader, normalizer, statistics, split = make_data(args, tokenizer, images, rank, world_size)
        splits = gather_report(split)
        metadata = build_formal_metadata(source_splits=splits, statistics=statistics, max_steps=args.max_steps,
            dataset_key=args.dataset_key,
            world_size=world_size, local_batch_size=args.per_device_batch_size,
            base_learning_rate=args.learning_rate, lr_decay_step=args.lr_decay_step,
            flow_head_learning_rate=args.flow_head_learning_rate,
            lr_decay_factor=args.lr_decay_factor, image_aug=args.image_aug, shuffle_buffer_size=args.shuffle_buffer_size,
            save_every=args.save_every, log_every=args.log_every, seed=args.seed)
        all_metadata = gather_report(metadata)
        require_all(all(value == metadata for value in all_metadata), "Ranks disagree on formal metadata/statistics", device)
        global_step = 0
        if args.resume is not None:
            error = None
            try:
                restored = load_hybrid_checkpoint(args.resume, ddp_model, optimizer, expected_metadata=metadata,
                    allow_max_steps_extension=args.allow_max_steps_extension)
                global_step = restored["global_step"]
                if global_step >= args.max_steps:
                    raise ValueError("Resume already completed: global_step must be smaller than requested max_steps")
                if optimizer_step_range(optimizer) != (global_step, global_step):
                    raise ValueError("Restored AdamW state steps do not match global_step")
            except Exception as exc:
                error = f"rank {rank}: {type(exc).__name__}: {exc}"
            errors = gather_report(error)
            if any(errors):
                raise RuntimeError(f"Formal resume rejected before training: {errors}")
            state_checks(model, optimizer, global_step, device, policy)
        rank_zero_io(lambda: prepare_run_manifest(args.run_dir, metadata, resume=args.resume is not None,
            allow_max_steps_extension=args.allow_max_steps_extension))
        segment_seed = args.seed + 1000 + rank + global_step
        torch.manual_seed(segment_seed)
        torch.cuda.manual_seed_all(segment_seed)
        torch.cuda.reset_peak_memory_stats(device)

        def checkpoint(step):
            state_checks(model, optimizer, step, device, policy)
            rank_zero_io(lambda: save_hybrid_checkpoint(args.run_dir / "checkpoints" / checkpoint_name(step),
                ddp_model, optimizer, global_step=step, metadata=metadata))

        global_step = training_loop(ddp_model, loader, normalizer, optimizer, args=args, global_step=global_step,
            log_callback=lambda step, lr, diag: write_log(args.run_dir, step, lr, diag, device), checkpoint_callback=checkpoint)
        require_all(global_step == args.max_steps and optimizer_step_range(optimizer) == (args.max_steps, args.max_steps),
                    "Final optimizer/global step mismatch", device)
        def verify_final():
            if not (args.run_dir / "checkpoints" / checkpoint_name(global_step)).is_file():
                raise RuntimeError("Final checkpoint is missing")
        rank_zero_io(verify_final)
        if rank == 0:
            print(f"HYBRID FORMAL TRAINING COMPLETED: dataset_key={args.dataset_key}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
