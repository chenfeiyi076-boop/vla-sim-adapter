# Phase 6A: Hybrid RLDS source sharding

`HybridRLDSDataset(..., rank=0, world_size=1)` accepts explicit keyword-only
rank configuration. It never queries torch.distributed. Both the Hybrid
constructor and RLDS split resolver reject invalid ranks, non-integer/bool
values, invalid sizes, and partial configuration. Lower-level callers may
omit both `shard_rank` and `shard_world_size` to retain legacy behavior.

For multiple ranks, `_resolve_rlds_split` uses
`tfds.even_splits("train" if train else "val", n=world_size, drop_remainder=False)`
and selects the rank's instruction. World size 1 preserves the exact original
`"train"`/`"val"` strings. These splits partition source episodes, without
manually specified percentage boundaries or dependence on local shuffle order.

The interleaver's first statistics/size pass remains unsharded. Its second
source construction pass forwards the rank options to `make_dataset_from_rlds`,
which supplies the instruction to `DLataset.from_rlds(split=...)`. This is
before repeat, H=10 window generation, flatten, frame interleaving, shuffle,
image decode/resize, and PyTorch iteration. No late `.shard()`, modulo/islice
filter or DistributedSampler is used: those would partition an already random
repeated stream and would not provide the required source-disjoint contract.

Statistics still use `from_rlds(split="all", shuffle=False)` with the existing
cache dependencies. They remain GLOBAL; no per-rank mean/std is computed and
the existing cached-statistics logic and HybridZScoreNormalizer are unchanged.
`dataset_length` remains the original GLOBAL effective-length estimate, not an
exact rank-local transition count. Episode lengths vary, and the interleaver
repeats and samples sources indefinitely. Phase 6A does not define epochs or
optimizer-step scheduling; Phase 6B will define an explicit step policy.

Use `DataLoader(..., num_workers=0)`. Hybrid iteration fails immediately inside
a non-main DataLoader worker because worker-level sharding is not implemented.
The generic legacy RLDSDataset worker behavior is unchanged. The existing
Phase-5B DDP mechanics smoke is also unchanged: it still constructs default
unsharded datasets; future training callers must explicitly pass rank/size.

## Real data-only server smoke

From the repo root in the existing server environment:

```sh
torchrun --standalone --nnodes=1 --nproc_per_node=4 vla-scripts/hybrid_sharding_smoke.py --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero --samples-per-rank 32
```

This uses four Gloo workers, hides CUDA devices, and loads only the existing
tokenizer/image processor. No VLM/flow weights, optimizer, backward, or CUDA
model memory is used. Each rank constructs its explicit episode source split
and consumes 32 real LIBERO-Spatial batches (optionally 64), batch size 1,
num_workers 0, image_aug False. The smoke uses a small 100-frame shuffle buffer.
Canonical actions/proprio are fingerprinted before any z-score normalization.

SHA-256 fingerprints include field names, shapes, dtypes, input IDs, all H10
actions, proprio, and both transformed camera views. Rank zero reports resolved
split instructions, samples obtained, within-rank duplicates, all pairwise
cross-rank intersection counts, the global statistics key and H/A/P dimensions.
PASS requires all four ranks to finish, four distinct source splits, valid
H=10/A=7/P=8 batches, the requested sample count, and zero sampled cross-rank
fingerprint overlap. Within-rank duplicates are reported but do not alone fail
the smoke, since repeated streams may revisit data. Failures produce rank error
reports and a nonzero exit, without printing the success line.

The exact success line is `PHASE 6A REAL RLDS SHARDING SMOKE PASSED`.
The fingerprint check is an empirical sanity check of a small sampled subset,
not proof that all dataset contents differ. Source disjointness comes from
TFDS episode split instructions. Distinct source episodes can theoretically
contain identical sample content; a sampled overlap must be investigated.
CPU fake-based unit tests do not replace this real TFDS/dlimp server run.
