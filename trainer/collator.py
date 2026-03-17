import dataclasses
import importlib.resources as pkg_resources
import json
import random
import copy
import warnings
import os
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from accelerate import Accelerator, PartialState
from accelerate.state import AcceleratorState
from huggingface_hub import ModelCard, ModelCardData
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    ProcessorMixin
)
from datasets import Dataset, IterableDataset
from qwen_omni_utils.v2_5 import process_mm_info


class PreprocessedMultimodalDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that performs multimodal preprocessing (audio/image/video loading)
    in __getitem__, enabling multi-process data loading via DataLoader workers.

    This moves the I/O-intensive operations (file reading) from the collator to the
    dataset, allowing them to be parallelized across multiple workers.
    """

    def __init__(self, dataset: Dataset, use_audio_in_video: bool = True, timeout: int = 30):
        self.dataset = dataset
        self.use_audio_in_video = use_audio_in_video
        self.timeout = timeout

    def __len__(self):
        return len(self.dataset)

    def _clean_conversation(self, conv):
        """Clean None values from conversation content. Returns a deep copy."""
        conv = copy.deepcopy(conv)
        for turn in conv:
            if turn.get('content') and isinstance(turn['content'], list):
                turn['content'] = [
                    {k: v for k, v in item.items() if v is not None}
                    for item in turn['content']
                ]
        return conv

    def __getitem__(self, idx):
        item = self.dataset[idx]

        chosen_conv = self._clean_conversation(item["chosen"])
        rejected_conv = self._clean_conversation(item["rejected"])

        # Perform I/O-intensive multimodal processing here (parallelized by DataLoader workers)
        audios_chosen, images_chosen, videos_chosen = process_mm_info(
            [chosen_conv], use_audio_in_video=self.use_audio_in_video
        )
        audios_rejected, images_rejected, videos_rejected = process_mm_info(
            [rejected_conv], use_audio_in_video=self.use_audio_in_video
        )

        result = {
            "chosen_conv": chosen_conv,
            "rejected_conv": rejected_conv,
            "audios_chosen": audios_chosen,
            "images_chosen": images_chosen,
            "videos_chosen": videos_chosen,
            "audios_rejected": audios_rejected,
            "images_rejected": images_rejected,
            "videos_rejected": videos_rejected,
        }

        if "margin" in item:
            result["margin"] = item["margin"]

        return result

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Processing timed out after {self.timeout} seconds")


@dataclass
class MultimodalRewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        processor (`ProcessorMixin`):
            The processor used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the processor.tokenizer.
        pad_to_multiple_of (`int` or `None`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    processor: ProcessorMixin
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]

        # Fast path: dataset already tokenized + padded offline
        is_tokenized = "input_ids_chosen" in features[0] and "input_ids_rejected" in features[0]
        if is_tokenized:
            tokenizer = getattr(self.processor, "tokenizer", None)
            pad_token_id = None
            if tokenizer is not None:
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            if pad_token_id is None and hasattr(self.processor, "pad_token_id"):
                pad_token_id = getattr(self.processor, "pad_token_id")
            pad_token_id = 0 if pad_token_id is None else pad_token_id

            def pad_and_stack(key: str, pad_value: float | int = 0):
                """Pad tensors in this key to the max shape and stack."""
                values = [f[key] for f in features if key in f and f[key] is not None]
                if not values:
                    return None
                tensors = [torch.tensor(v) for v in values]
                first_dim = tensors[0].dim()
                if any(t.dim() != first_dim for t in tensors):
                    raise ValueError(f"Mismatched dims for key {key} in pretokenized batch")
                max_shape = [0] * first_dim
                for t in tensors:
                    for i, size in enumerate(t.shape):
                        max_shape[i] = max(max_shape[i], size)
                padded = []
                for t in tensors:
                    pad_sizes = []
                    for i in reversed(range(first_dim)):
                        pad_sizes.extend([0, max_shape[i] - t.shape[i]])
                    pad_val = float(pad_value) if t.is_floating_point() else int(pad_value)
                    padded.append(F.pad(t, pad_sizes, value=pad_val))
                return torch.stack(padded)

            batch = {
                "input_ids_chosen": pad_and_stack("input_ids_chosen", pad_token_id),
                "attention_mask_chosen": pad_and_stack("attention_mask_chosen", 0),
                "input_features_chosen": pad_and_stack("input_features_chosen", 0.0),
                "feature_attention_mask_chosen": pad_and_stack("feature_attention_mask_chosen", 0),
                "pixel_values_chosen": pad_and_stack("pixel_values_chosen", 0.0),
                "image_grid_thw_chosen": pad_and_stack("image_grid_thw_chosen", 0),
                "pixel_values_videos_chosen": pad_and_stack("pixel_values_videos_chosen", 0.0),
                "video_grid_thw_chosen": pad_and_stack("video_grid_thw_chosen", 0),
                "video_second_per_grid_chosen": pad_and_stack("video_second_per_grid_chosen", 0.0),
                "input_ids_rejected": pad_and_stack("input_ids_rejected", pad_token_id),
                "attention_mask_rejected": pad_and_stack("attention_mask_rejected", 0),
                "input_features_rejected": pad_and_stack("input_features_rejected", 0.0),
                "feature_attention_mask_rejected": pad_and_stack("feature_attention_mask_rejected", 0),
                "pixel_values_rejected": pad_and_stack("pixel_values_rejected", 0.0),
                "image_grid_thw_rejected": pad_and_stack("image_grid_thw_rejected", 0),
                "pixel_values_videos_rejected": pad_and_stack("pixel_values_videos_rejected", 0.0),
                "video_grid_thw_rejected": pad_and_stack("video_grid_thw_rejected", 0),
                "video_second_per_grid_rejected": pad_and_stack("video_second_per_grid_rejected", 0.0),
                "return_loss": True,
            }

            if has_margin:
                batch["margin"] = torch.tensor([f["margin"] for f in features], dtype=torch.float)

            return batch

        # Check if data comes from PreprocessedMultimodalDataset (preprocessed)
        # or raw dataset (needs processing)
        is_preprocessed = "chosen_conv" in features[0]

        if is_preprocessed:
            # Data already preprocessed by PreprocessedMultimodalDataset
            # Just need to merge and apply processor for tokenization + padding
            batch_chosen_conv = [f["chosen_conv"] for f in features]
            batch_rejected_conv = [f["rejected_conv"] for f in features]

            # Merge preprocessed multimodal data from all samples
            audios_chosen = self._merge_multimodal_list([f["audios_chosen"] for f in features])
            images_chosen = self._merge_multimodal_list([f["images_chosen"] for f in features])
            videos_chosen = self._merge_multimodal_list([f["videos_chosen"] for f in features])
            audios_rejected = self._merge_multimodal_list([f["audios_rejected"] for f in features])
            images_rejected = self._merge_multimodal_list([f["images_rejected"] for f in features])
            videos_rejected = self._merge_multimodal_list([f["videos_rejected"] for f in features])

        elif "chosen" in features[0] and "rejected" in features[0]:
            # Raw data - process from scratch (original behavior, single-threaded)
            batch_chosen_conv = []
            batch_rejected_conv = []
            for feature in features:
                chosen_conv = feature["chosen"]
                rejected_conv = feature["rejected"]
                # Clean None values
                for conv in [chosen_conv, rejected_conv]:
                    for turn in conv:
                        if turn.get('content') and isinstance(turn['content'], list):
                            turn['content'] = [
                                {k: v for k, v in item.items() if v is not None}
                                for item in turn['content']
                            ]
                batch_chosen_conv.append(chosen_conv)
                batch_rejected_conv.append(rejected_conv)

            audios_chosen, images_chosen, videos_chosen = process_mm_info(
                batch_chosen_conv, use_audio_in_video=True
            )
            audios_rejected, images_rejected, videos_rejected = process_mm_info(
                batch_rejected_conv, use_audio_in_video=True
            )
        else:
            raise ValueError("The features should include `chosen` and `rejected` conversations "
                           "or preprocessed `chosen_conv` and `rejected_conv`.")

        # Apply processor for tokenization and padding
        text_chosen = self.processor.apply_chat_template(
            batch_chosen_conv, add_generation_prompt=False, tokenize=False
        )
        batch_chosen = self.processor(
            text=text_chosen, audio=audios_chosen, images=images_chosen,
            videos=videos_chosen, return_tensors="pt", padding=True, use_audio_in_video=True
        )

        text_rejected = self.processor.apply_chat_template(
            batch_rejected_conv, add_generation_prompt=False, tokenize=False
        )
        batch_rejected = self.processor(
            text=text_rejected, audio=audios_rejected, images=images_rejected,
            videos=videos_rejected, return_tensors="pt", padding=True, use_audio_in_video=True
        )

        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_features_chosen": batch_chosen.get("input_features", None),
            "feature_attention_mask_chosen": batch_chosen.get("feature_attention_mask", None),
            "pixel_values_chosen": batch_chosen.get("pixel_values", None),
            "image_grid_thw_chosen": batch_chosen.get("image_grid_thw", None),
            "pixel_values_videos_chosen": batch_chosen.get("pixel_values_videos", None),
            "video_grid_thw_chosen": batch_chosen.get("video_grid_thw", None),
            "video_second_per_grid_chosen": batch_chosen.get("video_second_per_grid", None),
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "input_features_rejected": batch_rejected.get("input_features", None),
            "feature_attention_mask_rejected": batch_rejected.get("feature_attention_mask", None),
            "pixel_values_rejected": batch_rejected.get("pixel_values", None),
            "image_grid_thw_rejected": batch_rejected.get("image_grid_thw", None),
            "pixel_values_videos_rejected": batch_rejected.get("pixel_values_videos", None),
            "video_grid_thw_rejected": batch_rejected.get("video_grid_thw", None),
            "video_second_per_grid_rejected": batch_rejected.get("video_second_per_grid", None),
            "return_loss": True,
        }

        if has_margin:
            margin = torch.tensor([f["margin"] for f in features], dtype=torch.float)
            batch["margin"] = margin

        return batch

    def _merge_multimodal_list(self, items_list: list) -> list | None:
        """Merge multimodal data from multiple samples into a single list."""
        merged = []
        for items in items_list:
            if items is not None:
                merged.extend(items)
        return merged if merged else None
