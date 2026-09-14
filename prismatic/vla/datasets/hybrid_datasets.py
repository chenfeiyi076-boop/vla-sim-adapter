"""Opt-in H=10 bridge using canonical OXE values, before legacy normalization.

Use the existing Prismatic image processor (224px) and tokenizer. With RLDS,
use DataLoader num_workers=0 as in the legacy path. Normalization is explicit
and separate: HybridZScoreNormalizer(dataset.dataset_statistics[dataset_name]).
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import get_worker_info

from prismatic.models.backbones.llm.prompting import QwenPromptBuilder
from prismatic.vla.datasets.datasets import RLDSDataset
from prismatic.vla.datasets.rlds.dataset import _resolve_rlds_split


HYBRID_ACTION_HORIZON = 10
HYBRID_ACTION_DIM = 7
HYBRID_PROPRIO_DIM = 8


class HybridRLDSDataset(RLDSDataset):
    """Same constructor as RLDSDataset; pass libero_spatial_no_noops as data_mix.

    The parent's configuration already loads primary/wrist, proprio, language,
    and the original OXE standardization. Override only the construction hook,
    before either pass through make_dataset_from_rlds in the interleaver.
    Inherited dataset_statistics and dataset_name retain their original meaning.
    dataset_length remains the GLOBAL effective-length estimate, not a rank-local
    cardinality. Rank sharding is explicit; worker-level sharding is unsupported.
    """

    def __init__(self, data_root_dir, data_mix, batch_transform, resize_resolution,
                 shuffle_buffer_size=256_000, train=True, image_aug=False, *, rank=0, world_size=1):
        if rank is None or world_size is None:
            raise ValueError("Hybrid rank and world_size must be explicit integers (defaults 0 and 1)")
        self.resolved_source_split = _resolve_rlds_split(
            train=train, shard_rank=rank, shard_world_size=world_size
        )
        self.rank, self.world_size = rank, world_size
        super().__init__(data_root_dir, data_mix, batch_transform, resize_resolution,
                         shuffle_buffer_size=shuffle_buffer_size, train=train, image_aug=image_aug)

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("Hybrid RLDS worker-level sharding is not implemented; num_workers=0 is required")
        return super().__iter__()

    def make_dataset(self, rlds_config):
        config = deepcopy(rlds_config)
        config.update(shard_rank=self.rank, shard_world_size=self.world_size)
        config["traj_transform_kwargs"].update(
            window_size=1, future_action_window_size=HYBRID_ACTION_HORIZON - 1
        )
        for kwargs in config["dataset_kwargs_list"]:
            kwargs["normalize_action_proprio"] = False
        return super().make_dataset(config)


@dataclass
class HybridRLDSBatchTransform:
    """A canonical RLDS frame -> observation-only prompt and unnormalized tensors."""

    base_tokenizer: Callable
    image_transform: Callable

    def __call__(self, rlds_batch):
        actions = torch.as_tensor(np.asarray(rlds_batch["action"]).copy(), dtype=torch.float32)
        if actions.shape != (HYBRID_ACTION_HORIZON, HYBRID_ACTION_DIM):
            raise ValueError(f"Hybrid actions must be [10,7], got {tuple(actions.shape)}")
        observation = rlds_batch["observation"]
        proprio = torch.as_tensor(np.asarray(observation["proprio"]).copy(), dtype=torch.float32)
        if proprio.shape != (1, HYBRID_PROPRIO_DIM):
            raise ValueError(f"RLDS current proprio window must be [1,8], got {tuple(proprio.shape)}")

        language = rlds_batch["task"]["language_instruction"]
        if isinstance(language, bytes):
            language = language.decode("utf-8")
        if not isinstance(language, str):
            raise ValueError("language_instruction must be a UTF-8 string or bytes")
        builder = QwenPromptBuilder("openvla")
        builder.add_turn("human", f"What action should the robot take to {language.lower()}?")
        # One human turn leaves the existing builder at the assistant prefix.
        # No response, EOS suffix, action tokens, token deletion, or labels.
        ids = self.base_tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids

        pixels = {}
        for source, target in (("image_primary", "pixel_values"), ("image_wrist", "pixel_values_wrist")):
            image = np.asarray(observation[source])
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
                raise ValueError(f"{source} must be a single RGB observation window [1,H,W,3]")
            pixels[target] = self.image_transform(Image.fromarray(image[0]))
        return dict(
            input_ids=torch.as_tensor(ids, dtype=torch.long),
            **pixels,
            actions=actions,
            proprio=proprio[0],
            dataset_name=rlds_batch["dataset_name"],
        )


@dataclass
class PaddedCollatorForHybridFlow:
    """Right pad by sequence length; stack [primary channels, wrist channels]."""

    pad_token_id: int

    def __call__(self, instances: Sequence[dict]) -> dict:
        if not instances:
            raise ValueError("Cannot collate an empty Hybrid batch")
        for sample in instances:
            if sample["input_ids"].ndim != 1 or sample["input_ids"].numel() == 0:
                raise ValueError("input_ids must be a nonempty [L] tensor")
            if sample["actions"].shape != (HYBRID_ACTION_HORIZON, HYBRID_ACTION_DIM):
                raise ValueError("Hybrid actions must have shape [10,7]")
            if sample["proprio"].shape != (HYBRID_PROPRIO_DIM,):
                raise ValueError("Hybrid proprio must have shape [8]")
            for key in ("pixel_values", "pixel_values_wrist"):
                if not isinstance(sample[key], torch.Tensor) or sample[key].ndim != 3:
                    raise ValueError(f"{key} must be a transformed [C,H,W] tensor")
                if sample[key].shape != instances[0]["pixel_values"].shape:
                    raise ValueError("All primary/wrist tensors must share the same [C,H,W] shape")
        sequences = [sample["input_ids"] for sample in instances]
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id)
        lengths = torch.tensor([len(ids) for ids in sequences], device=input_ids.device)
        # True = actual input token, including any occurrence of pad_token_id
        # inside a supplied sequence; only newly appended padding is False.
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None] < lengths[:, None]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=torch.cat([
                torch.stack([sample["pixel_values"] for sample in instances]),
                torch.stack([sample["pixel_values_wrist"] for sample in instances]),
            ], dim=1),
            actions=torch.stack([sample["actions"] for sample in instances]),
            proprio=torch.stack([sample["proprio"] for sample in instances]),
            dataset_names=[sample["dataset_name"] for sample in instances],
        )
