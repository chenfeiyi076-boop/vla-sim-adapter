"""Phase 6C: separate four-GPU save (0->5) and process-restart resume (5->10) jobs."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def representatives(model):
    return {"encoder": model.encoder.projector.fc1.weight, "flow_head": model.flow_head.action_decoder.weight}


def state_checks(model, optimizer, global_step, device, policy):
    from hybrid_ddp_smoke import require_all, synchronized_slice
    from hybrid_multistep_smoke import optimizer_step_range

    low, high = optimizer_step_range(optimizer)
    require_all(low == high == global_step, "AdamW state steps do not match global_step", device)
    require_all({name: p.requires_grad for name, p in model.named_parameters()} == policy
                and all(not p.requires_grad and p.grad is None for p in model.encoder.action_queries.parameters())
                and all(p.grad is None for p in model.parameters() if not p.requires_grad),
                "Excluded parameter policy changed", device)
    checks = dict(optimizer_state_step_min=low, optimizer_state_step_max=high)
    for name, p in representatives(model).items():
        checks[name + "_parameter_max_diff"] = synchronized_slice(p, name + " parameter", device)
        for field in ("exp_avg", "exp_avg_sq"):
            checks[f"{name}_{field}_max_diff"] = synchronized_slice(optimizer.state[p][field], f"{name} {field}", device)
    checks["representative_optimizer_state_synced"] = True
    return checks


def train_segment(ddp_model, loader, normalizer, optimizer, global_step, device):
    from hybrid_ddp_smoke import require_all, synchronized_slice
    from prismatic.training.hybrid_multistep import hybrid_training_step

    params = representatives(ddp_model.module)  # Inspection only; forward remains inside DDP.
    initial = {name: p.detach().reshape(-1)[:64].clone() for name, p in params.items()}
    iterator = iter(loader)
    records = []
    for _ in range(5):
        batch = next(iterator)
        if batch["actions"].shape != (1, 10, 7) or batch["proprio"].shape != (1, 8):
            raise ValueError("Expected one H10/A7/P8 sample per rank")
        result = hybrid_training_step(ddp_model, batch, normalizer, optimizer, device_type="cuda")
        global_step += 1  # Only after the successful optimizer step.
        result["global_step"] = global_step
        for name, p in params.items():
            result[name + "_gradient_max_diff"] = synchronized_slice(p.grad, name + " gradient", device)
            result[name + "_parameter_max_diff"] = synchronized_slice(p, name + " parameter", device)
        optimizer.zero_grad(set_to_none=True)
        records.append(result)
        del batch, result
    changed = {name + "_changed": not torch.equal(p.detach().reshape(-1)[:64], initial[name]) for name, p in params.items()}
    require_all(all(changed.values()), "Encoder/flow representative did not change during segment", device)
    return global_step, dict(rank=dist.get_rank(), steps=records, **changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("save", "resume"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5, choices=(5,))
    args = parser.parse_args()
    if not all(path.is_dir() for path in (args.vlm_path, args.hf_config, args.data_root)):
        parser.error("Native VLM, HF config and RLDS data directories must exist")
    if args.mode == "save" and os.path.lexists(args.checkpoint):
        parser.error("Checkpoint already exists; choose a new path")
    if args.mode == "resume" and not args.checkpoint.is_file():
        parser.error("Resume requires the existing Run-A checkpoint")
    if int(os.environ["WORLD_SIZE"]) != 4 or not torch.cuda.is_available():
        raise RuntimeError("Use a separate torchrun job with exactly four CUDA workers")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=15))
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        from hybrid_ddp_smoke import load_assets, gather_report, require_all
        from hybrid_multistep_smoke import make_data
        from prismatic.training.hybrid_checkpoint import save_hybrid_checkpoint, load_hybrid_checkpoint
        from prismatic.training.hybrid_ddp import HybridFlowTrainingModule
        from prismatic.training.hybrid_step import hybrid_parameter_groups

        rank = dist.get_rank()
        torch.manual_seed(7)
        encoder, head, tokenizer, images = load_assets(args.vlm_path, args.hf_config)
        model = HybridFlowTrainingModule(encoder, head).to(device)
        groups = hybrid_parameter_groups(model.encoder, model.flow_head)
        policy = {name: p.requires_grad for name, p in model.named_parameters()}
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(groups, lr=1e-6)
        if args.mode == "save":
            torch.manual_seed(1000 + rank)
            torch.cuda.manual_seed_all(1000 + rank)
        loader, normalizer, split = make_data(args.data_root, tokenizer, images, rank, dist.get_world_size())
        splits = gather_report(split)
        require_all(len(splits) == 4 and len(set(splits)) == 4, "Source splits must be four distinct instructions", device)
        metadata = dict(dataset_key="libero_spatial_no_noops", world_size=4, local_batch_size=1,
                        effective_global_batch_size=4, action_horizon=10, action_dim=7, proprio_dim=8,
                        optimizer="AdamW", learning_rate=1e-6, gradient_accumulation=1,
                        scheduler=None, source_splits=splits)
        global_step = 0
        pretraining = None
        if args.mode == "resume":
            error = None
            try:
                loaded = load_hybrid_checkpoint(args.checkpoint, ddp_model, optimizer, expected_metadata=metadata)
                if loaded["global_step"] != 5:
                    raise ValueError("Formal resume requires saved global_step == 5")
            except Exception as exc:
                error = f"rank {rank}: {type(exc).__name__}: {exc}"
            errors = gather_report(error)
            if any(errors):
                raise RuntimeError(f"Resume load failed before training: {errors}")
            global_step = loaded["global_step"]
            pretraining = state_checks(model, optimizer, global_step, device, policy)
            torch.manual_seed(2000 + rank)
            torch.cuda.manual_seed_all(2000 + rank)
        start = global_step
        global_step, local = train_segment(ddp_model, loader, normalizer, optimizer, global_step, device)
        require_all(global_step == (5 if args.mode == "save" else 10), "Wrong end global_step", device)
        checks = state_checks(model, optimizer, global_step, device, policy)
        local.update(checks)
        local["pretraining_load_checks"] = pretraining
        ranks = gather_report(local)
        report = dict(mode=args.mode, world_size=4, steps_this_run=5, start_global_step=start,
                      end_global_step=global_step, source_splits=splits, source_splits_distinct=True,
                      ranks=ranks, **checks)
        if args.mode == "save":
            status = [None]
            if rank == 0:
                try:
                    save_hybrid_checkpoint(args.checkpoint, ddp_model, optimizer, global_step=global_step, metadata=metadata)
                    status[0] = dict(error=None, size=args.checkpoint.stat().st_size)
                except Exception as exc:
                    status[0] = dict(error=f"{type(exc).__name__}: {exc}")
            dist.broadcast_object_list(status, src=0)
            if status[0]["error"] is not None:
                raise RuntimeError(f"Rank-zero checkpoint save failed: {status[0]['error']}")
            report.update(checkpoint=str(args.checkpoint), checkpoint_exists=True,
                          checkpoint_size_bytes=status[0]["size"], save_passed=True)
        else:
            report.update(loaded_format_version=loaded["format_version"], loaded_global_step=loaded["global_step"],
                          source_splits_match_checkpoint=True, resume_passed=True)
        if rank == 0:
            print(json.dumps(report, indent=2), flush=True)
            print(f"PHASE 6C CHECKPOINT {args.mode.upper()} PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
