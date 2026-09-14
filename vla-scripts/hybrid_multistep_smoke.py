"""Phase 6B: 20 continuous source-sharded RLDS/DDP steps on four CUDA GPUs."""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_data(data_root, tokenizer, images, rank, world_size):
    from torch.utils.data import DataLoader
    from prismatic.vla.datasets.hybrid_datasets import (
        HybridRLDSDataset, HybridRLDSBatchTransform, PaddedCollatorForHybridFlow,
    )
    from prismatic.vla.hybrid_normalization import HybridZScoreNormalizer

    dataset = HybridRLDSDataset(
        data_root, "libero_spatial_no_noops", HybridRLDSBatchTransform(tokenizer, images.apply_transform),
        resize_resolution=(224, 224), shuffle_buffer_size=100, train=True, image_aug=False,
        rank=rank, world_size=world_size,
    )
    normalizer = HybridZScoreNormalizer(dataset.dataset_statistics["libero_spatial_no_noops"])
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=PaddedCollatorForHybridFlow(tokenizer.pad_token_id))
    return loader, normalizer, str(dataset.resolved_source_split)


def optimizer_step_range(optimizer):
    """Inspect every AdamW state; missing states/steps fail rather than being skipped."""
    steps = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if state is None or "step" not in state:
                raise RuntimeError("Missing AdamW state step for a selected parameter")
            steps.append(float(state["step"]))
    if not steps or not all(math.isfinite(step) for step in steps):
        raise RuntimeError("Invalid AdamW state step")
    return min(steps), max(steps)


def memory_span(allocated):
    steady = allocated[3:]
    return max(steady) - min(steady) if steady else None


def run_steps(ddp_model, loader, normalizer, optimizer, device, steps):
    from hybrid_ddp_smoke import synchronized_slice
    from prismatic.training.hybrid_multistep import hybrid_training_step

    model = ddp_model.module  # Inspection only; forward is owned by hybrid_training_step.
    representatives = {"encoder": model.encoder.projector.fc1.weight,
                       "flow_head": model.flow_head.action_decoder.weight}
    initial = {name: p.detach().reshape(-1)[:64].clone() for name, p in representatives.items()}
    iterator = iter(loader)
    records, sync_records, allocated, peaks = [], [], [], []
    for global_step in range(1, steps + 1):
        batch = next(iterator)
        if (batch["actions"].shape != (1, 10, 7) or batch["proprio"].shape != (1, 8)
                or batch["input_ids"].shape[0] != 1 or batch["pixel_values"].shape[0] != 1):
            raise ValueError("Each rank must consume one H10/A7/P8 sample per step")
        torch.cuda.reset_peak_memory_stats(device)
        diagnostics = hybrid_training_step(ddp_model, batch, normalizer, optimizer, device_type="cuda",
                                           autocast_dtype=torch.bfloat16, autocast_enabled=True)
        # Logging only; DDP already reduced gradients during backward.
        mean = torch.tensor(diagnostics["loss"], device=device, dtype=torch.float64)
        dist.all_reduce(mean, op=dist.ReduceOp.SUM)
        mean /= dist.get_world_size()
        sync = {"global_step": global_step, "global_mean_loss": mean.item()}
        for name, p in representatives.items():
            sync[name + "_gradient_max_diff"] = synchronized_slice(p.grad, name + " gradient", device)
            sync[name + "_parameter_max_diff"] = synchronized_slice(p, name + " parameter", device)
        optimizer.zero_grad(set_to_none=True)
        records.append(diagnostics)  # Only Python scalars/flags, no autograd objects.
        sync_records.append(sync)
        del batch, diagnostics, mean, sync
        torch.cuda.synchronize(device)
        allocated.append(torch.cuda.memory_allocated(device) / 2**30)
        peaks.append(torch.cuda.max_memory_allocated(device) / 2**30)
    changed = {name + "_changed": not torch.equal(p.detach().reshape(-1)[:64], initial[name])
               for name, p in representatives.items()}
    low, high = optimizer_step_range(optimizer)
    report = dict(rank=dist.get_rank(), samples_consumed=len(records),
                  losses=[r["loss"] for r in records], t_means=[r["t_mean"] for r in records],
                  velocity_rms=[r["velocity_rms"] for r in records],
                  encoder_grad_norms=[r["encoder_grad_norm"] for r in records],
                  flow_head_grad_norms=[r["flow_head_grad_norm"] for r in records],
                  encoder_all_steps_finite_nonzero=all(math.isfinite(r["encoder_grad_norm"]) and r["encoder_grad_norm"] > 0 for r in records),
                  flow_head_all_steps_finite_nonzero=all(math.isfinite(r["flow_head_grad_norm"]) and r["flow_head_grad_norm"] > 0 for r in records),
                  all_steps_missing_trainable_grads=[n for r in records for n in r["missing_trainable_grads"]],
                  excluded_no_grad_all_steps=all(r["excluded_no_grad"] for r in records),
                  action_queries_no_grad_all_steps=all(r["action_queries_no_grad"] for r in records),
                  allocated_after_step_gib=allocated, peak_after_step_gib=peaks,
                  peak_cuda_allocated_gib=max(peaks), steady_memory_span_gib=memory_span(allocated),
                  optimizer_state_step_min=low, optimizer_state_step_max=high, **changed)
    return report, sync_records


def summarize(reports, sync_records, source_splits, steps):
    distinct = len(source_splits) == 4 and len(set(source_splits)) == 4
    good = len(reports) == 4 and distinct and len(sync_records) == steps
    for report in reports:
        good = good and (report["samples_consumed"] == steps
            and len(report["losses"]) == steps and all(math.isfinite(x) for x in report["losses"])
            and report["encoder_all_steps_finite_nonzero"] and report["flow_head_all_steps_finite_nonzero"]
            and not report["all_steps_missing_trainable_grads"] and report["excluded_no_grad_all_steps"]
            and report["action_queries_no_grad_all_steps"] and report["encoder_changed"] and report["flow_head_changed"]
            and report["optimizer_state_step_min"] == steps and report["optimizer_state_step_max"] == steps
            and (report["steady_memory_span_gib"] is None if steps <= 3 else
                 report["steady_memory_span_gib"] is not None and report["steady_memory_span_gib"] <= .5))
    for record in sync_records:
        good = good and math.isfinite(record["global_mean_loss"]) and all(
            math.isfinite(record[key]) and record[key] <= 1e-7
            for key in ("encoder_gradient_max_diff", "flow_head_gradient_max_diff",
                        "encoder_parameter_max_diff", "flow_head_parameter_max_diff"))
    return dict(world_size=4, optimizer_steps=steps, local_batch_size=1, effective_global_batch_size=4,
                scheduler=None, gradient_accumulation=1, source_splits=source_splits,
                source_splits_distinct=distinct, steps=sync_records, ranks=reports,
                optimizer_state_step_min=min(r["optimizer_state_step_min"] for r in reports),
                optimizer_state_step_max=max(r["optimizer_state_step_max"] for r in reports),
                requested_run_passed=bool(good), phase6b_passed=bool(good and steps == 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/x2227/datasets/libero"))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        parser.error("--steps must be 1..20; formal Phase-6B acceptance requires 20")
    if not all(p.is_dir() for p in (args.vlm_path, args.hf_config, args.data_root)):
        parser.error("Native VLM, HF config and data directories must exist")
    if int(os.environ["WORLD_SIZE"]) != 4 or not torch.cuda.is_available():
        raise RuntimeError("Use torchrun with exactly four CUDA GPU workers")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=15))
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        from hybrid_ddp_smoke import load_assets, gather_report
        from prismatic.training.hybrid_ddp import HybridFlowTrainingModule
        from prismatic.training.hybrid_step import hybrid_parameter_groups

        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(7)
        encoder, flow_head, tokenizer, images = load_assets(args.vlm_path, args.hf_config)
        model = HybridFlowTrainingModule(encoder, flow_head).to(device)
        groups = hybrid_parameter_groups(model.encoder, model.flow_head)
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(groups, lr=1e-6)
        seed = 1000 + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        loader, normalizer, split = make_data(args.data_root, tokenizer, images, rank, world_size)
        splits = gather_report(split)
        if len(splits) != 4 or len(set(splits)) != 4:
            raise RuntimeError(f"Expected four distinct Phase-6A source splits, got {splits}")
        local_report, sync_records = run_steps(ddp_model, loader, normalizer, optimizer, device, args.steps)
        reports = gather_report(local_report)
        result = summarize(reports, sync_records, splits, args.steps)
        if rank == 0:
            # Print full memory series before failing; never raise the 0.5 GiB threshold.
            print(json.dumps(result, indent=2), flush=True)
        if not result["requested_run_passed"]:
            raise RuntimeError("Phase 6B checks failed; inspect rank-zero report including per-step memory")
        if rank == 0:
            if result["phase6b_passed"]:
                print("PHASE 6B REAL MULTI-STEP DDP TRAINING SMOKE PASSED", flush=True)
            else:
                print("Debug run completed; formal Phase-6B acceptance requires 20 steps.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
