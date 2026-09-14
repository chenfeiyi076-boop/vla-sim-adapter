"""Exactly one real RLDS batch/rank and AdamW step on 4 CUDA GPUs via torchrun.

This validates DDP mechanics, NOT long-run rank-aware IterableDataset sharding.
No simulator, Euler, evaluation, checkpoint or multi-step training loop.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# torchrun executes this file with vla-scripts as sys.path[0].
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def gather_report(report):
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report)
    return reports


def require_all(condition, message, device):
    flag = torch.tensor(int(bool(condition)), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError(message)


def synchronized_slice(value, name, device, atol=1e-7):
    """Compare a fixed first-64 slice against rank zero; do not reduce gradients manually."""
    local = value.detach().reshape(-1)[:64].float().contiguous()
    copies = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(copies, local)
    difference = max((other - copies[0]).abs().max().item() for other in copies)
    require_all(torch.isfinite(local).all() and difference <= atol,
                f"{name} is not synchronized (max diff={difference})", device)
    return difference


def check_gradients(model, groups, device):
    report = {}
    for group in groups:
        grads = [p.grad for p in group["params"] if p.grad is not None]
        finite = bool(grads) and all(torch.isfinite(g).all().item() for g in grads)
        nonzero = bool(grads) and any(torch.count_nonzero(g).item() for g in grads)
        report[group["name"] + "_finite_nonzero"] = finite and nonzero
    report["missing_trainable_grads"] = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    report["excluded_no_grad"] = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    report["action_queries_no_grad"] = all(p.grad is None for p in model.encoder.action_queries.parameters())
    reports = gather_report(report)
    good = all(r["encoder_finite_nonzero"] and r["flow_head_finite_nonzero"]
               and r["excluded_no_grad"] and r["action_queries_no_grad"]
               and not r["missing_trainable_grads"] for r in reports)
    if not good:
        raise RuntimeError("DDP gradient validation failed; do not change Phase-5A exclusions silently: " + json.dumps(reports))
    return report


def one_step(ddp_model, batch, normalizer, optimizer, groups, device):
    """Call outer DDP forward exactly once; inspect submodules only for diagnostics."""
    if batch["actions"].shape != (1, 10, 7) or batch["proprio"].shape != (1, 8):
        raise ValueError("Smoke requires batch size 1 per rank, H=10/A=7/P=8")
    model = ddp_model.module
    # Explicit representative tensors are consistent across ranks and small to snapshot.
    representatives = {"encoder": model.encoder.projector.fc1.weight,
                       "flow_head": model.flow_head.action_decoder.weight}
    before = {name: p.detach().reshape(-1)[:64].clone() for name, p in representatives.items()}
    optimizer.zero_grad(set_to_none=True)
    ddp_model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, diagnostics = ddp_model(batch, normalizer)
    require_all(torch.isfinite(loss).item(), "Non-finite local loss on at least one rank", device)
    loss.backward()
    report = check_gradients(model, groups, device)
    report.update(rank=dist.get_rank(), local_loss=loss.detach().item(), t_mean=diagnostics["t_mean"].item())
    for name, p in representatives.items():
        report[name + "_gradient_max_diff"] = synchronized_slice(p.grad, name + " gradient", device)
    optimizer.step()
    for name, p in representatives.items():
        changed = not torch.equal(p.detach().reshape(-1)[:64], before[name])
        require_all(changed, f"{name} representative parameter did not change on every rank", device)
        report[name + "_changed"] = changed
        report[name + "_parameter_max_diff"] = synchronized_slice(p, name + " parameter", device)
    torch.cuda.synchronize(device)
    report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    return gather_report(report)


def load_assets(vlm_path, hf_config):
    # Keep the validated native->HF mapping local: no evaluator import or saved HF weights.
    from transformers import AutoTokenizer
    from prismatic.models import load
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import PrismaticForConditionalGeneration
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
    from prismatic.models.flow_action_head import SimVLAFlowActionHead

    config = OpenVLAConfig.from_pretrained(hf_config, local_files_only=True)
    if (config.text_config.hidden_size != 896 or config.text_config.model_type != "qwen2"
            or config.vision_backbone_id != "dinosiglip-vit-so-224px" or config.image_sizes != [224, 224]):
        raise ValueError("Expected Prismatic DINOv2/SigLIP 224 + Qwen2.5-0.5B config")
    native = load(str(vlm_path), hf_token="", load_for_training=True)
    encoder = PrismaticForConditionalGeneration(config)
    replacements = (
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"), ("gamma", "scale_factor"),
    )
    converted = {}
    for key, value in native.state_dict().items():
        for old, new in replacements:
            key = key.replace(old, new)
        converted[key] = value
    missing, unexpected = encoder.load_state_dict(converted, strict=False)
    if set(missing) - {"action_queries.weight"} or unexpected:
        raise ValueError(f"Native weights mismatch: missing={missing}, unexpected={unexpected}")
    del native, converted
    encoder.vision_backbone.set_num_images_in_input(2)
    tokenizer = AutoTokenizer.from_pretrained(hf_config, local_files_only=True, trust_remote_code=False)
    image_processor = PrismaticImageProcessor.from_pretrained(hf_config, local_files_only=True)
    return encoder, SimVLAFlowActionHead(), tokenizer, image_processor


def real_batch(data_root, tokenizer, image_processor):
    from torch.utils.data import DataLoader
    from prismatic.vla.datasets.hybrid_datasets import (
        HybridRLDSDataset, HybridRLDSBatchTransform, PaddedCollatorForHybridFlow,
    )
    from prismatic.vla.hybrid_normalization import HybridZScoreNormalizer

    dataset = HybridRLDSDataset(
        data_root, "libero_spatial_no_noops",
        HybridRLDSBatchTransform(tokenizer, image_processor.apply_transform),
        resize_resolution=(224, 224), shuffle_buffer_size=100, train=True, image_aug=False,
    )
    normalizer = HybridZScoreNormalizer(dataset.dataset_statistics["libero_spatial_no_noops"])
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=PaddedCollatorForHybridFlow(tokenizer.pad_token_id))
    return next(iter(loader)), normalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/x2227/datasets/libero"))
    args = parser.parse_args()
    if not all(path.is_dir() for path in (args.vlm_path, args.hf_config, args.data_root)):
        parser.error("Native VLM, HF config and RLDS data directories must exist")
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or not torch.cuda.is_available():
        raise RuntimeError("Use torchrun with exactly four CUDA GPU workers")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=15))
    try:
        # RLDS decoding stays on CPU and must not reserve each process's GPU memory.
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        from prismatic.training.hybrid_ddp import HybridFlowTrainingModule
        from prismatic.training.hybrid_step import hybrid_parameter_groups

        torch.manual_seed(7)
        torch.cuda.reset_peak_memory_stats(device)
        encoder, head, tokenizer, images = load_assets(args.vlm_path, args.hf_config)
        model = HybridFlowTrainingModule(encoder, head).to(device)
        groups = hybrid_parameter_groups(model.encoder, model.flow_head)
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(groups, lr=1e-6)
        batch, normalizer = real_batch(args.data_root, tokenizer, images)
        # DDP's normal rank-zero parameter broadcast has finished. Each rank now
        # samples different t/noise, even if its independent RLDS iterator agrees.
        seed = 1000 + dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        reports = one_step(ddp_model, batch, normalizer, optimizer, groups, device)
        if dist.get_rank() == 0:
            print(json.dumps({"optimizer_steps": 1, "rank_sharding_validated": False, "ranks": reports}, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
