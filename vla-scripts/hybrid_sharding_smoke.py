"""Data-only four-rank Gloo sanity check of real Hybrid RLDS source splits."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DATASET_KEY = "libero_spatial_no_noops"


def fingerprint(batch):
    """Hash actual collated contents, with shape/dtype/field boundaries included."""
    shapes = {"actions": (1, 10, 7), "proprio": (1, 8), "pixel_values": (1, 12, 224, 224)}
    for key, shape in shapes.items():
        if tuple(batch[key].shape) != shape:
            raise ValueError(f"{key} must be {shape}, got {tuple(batch[key].shape)}")
    if batch["input_ids"].ndim != 2 or batch["input_ids"].shape[0] != 1 or batch["input_ids"].shape[1] == 0:
        raise ValueError("input_ids must be nonempty [1,L]")
    values = {"input_ids": batch["input_ids"], "actions": batch["actions"], "proprio": batch["proprio"],
              "primary_pixels": batch["pixel_values"][:, :6], "wrist_pixels": batch["pixel_values"][:, 6:]}
    digest = hashlib.sha256()
    for name, tensor in values.items():
        array = tensor.detach().cpu().contiguous().numpy()
        header = json.dumps([name, list(array.shape), str(array.dtype)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def summarize(reports, samples_per_rank):
    sets = [set(report["fingerprints"]) for report in reports]
    intersections = {f"{i}-{j}": len(sets[i] & sets[j])
                     for i in range(len(reports)) for j in range(i + 1, len(reports))}
    splits = [report["source_split"] for report in reports]
    passed = (len(reports) == 4 and {r["rank"] for r in reports} == set(range(4))
              and all(r["error"] is None and r["dimensions_valid"] and r["statistics_key"] == DATASET_KEY
                      and len(r["fingerprints"]) == samples_per_rank for r in reports)
              and None not in splits and len(set(splits)) == 4 and not any(intersections.values()))
    return dict(world_size=len(reports), samples_per_rank=samples_per_rank,
                ranks=[{key: value for key, value in r.items() if key != "fingerprints"} |
                       {"samples_obtained": len(r["fingerprints"]),
                        "within_rank_duplicate_count": len(r["fingerprints"]) - len(set(r["fingerprints"]))}
                       for r in reports],
                pairwise_cross_rank_intersections=intersections,
                global_statistics_key=DATASET_KEY, H=10, A=7, P=8, rank_sharding_validated=bool(passed))


def consume(args, rank, world_size):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
    from prismatic.vla.datasets.hybrid_datasets import (
        HybridRLDSDataset, HybridRLDSBatchTransform, PaddedCollatorForHybridFlow,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_config, local_files_only=True, trust_remote_code=False)
    images = PrismaticImageProcessor.from_pretrained(args.hf_config, local_files_only=True)
    dataset = HybridRLDSDataset(
        args.data_root, DATASET_KEY, HybridRLDSBatchTransform(tokenizer, images.apply_transform),
        resize_resolution=(224, 224), shuffle_buffer_size=100, train=True, image_aug=False,
        rank=rank, world_size=world_size,
    )
    if DATASET_KEY not in dataset.dataset_statistics:
        raise ValueError("Missing global libero_spatial_no_noops statistics")
    report = dict(rank=rank, source_split=str(dataset.resolved_source_split), statistics_key=DATASET_KEY,
                  dimensions_valid=True, fingerprints=[], error=None)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=PaddedCollatorForHybridFlow(tokenizer.pad_token_id))
    iterator = iter(loader)
    for _ in range(args.samples_per_rank):
        batch = next(iterator)
        report["fingerprints"].append(fingerprint(batch))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-config", type=Path, default=Path("pretrained_models/configs"))
    parser.add_argument("--data-root", type=Path, default=Path("/data/x2227/datasets/libero"))
    parser.add_argument("--samples-per-rank", type=int, choices=(32, 64), default=32)
    args = parser.parse_args()
    # This process has no GPU work, including TensorFlow dataset decoding.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import torch.distributed as dist

    if int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("Launch exactly four workers with torchrun")
    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=15))
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        try:
            if not args.hf_config.is_dir() or not args.data_root.is_dir():
                raise ValueError("HF config and RLDS data directories must exist")
            report = consume(args, rank, world_size)
        except Exception as error:
            report = dict(rank=rank, source_split=None, statistics_key=None,
                          dimensions_valid=False, fingerprints=[], error=f"{type(error).__name__}: {error}")
        reports = [None] * world_size if rank == 0 else None
        dist.gather_object(report, reports, dst=0)
        passed = torch.zeros((), dtype=torch.int64)
        if rank == 0:
            result = summarize(reports, args.samples_per_rank)
            print(json.dumps(result, indent=2), flush=True)
            passed.fill_(int(result["rank_sharding_validated"]))
        dist.broadcast(passed, src=0)
        if not passed.item():
            raise RuntimeError("Phase 6A real RLDS sharding smoke failed; inspect rank-zero report")
        if rank == 0:
            print("PHASE 6A REAL RLDS SHARDING SMOKE PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
