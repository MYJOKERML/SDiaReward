"""
inference.py - Clean inference example for SDiaReward

Usage:
    python inference.py \
        --ckpt_dir <path_to_reward_model_checkpoint> \
        --base_ckpt <path_to_base_qwen_omni_model> \
        --conversation_json <path_to_conversation.json>

The conversation JSON should be a list of message dicts, e.g.:
[
    {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "audio", "audio": "path/to/audio.wav"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}, {"type": "audio", "audio": "path/to/response.wav"}]}
]
"""
import argparse
import json
import torch
from transformers import Qwen2_5OmniThinkerConfig
from model.modeling_qwen_omni_thinker_reward import QwenOmniThinkerReward
from model.processing_qwen_omni_thinker_reward import OmniRewardProcessor
from qwen_omni_utils import process_mm_info


def parse_args():
    parser = argparse.ArgumentParser(description="SDiaReward inference example")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to the reward model checkpoint")
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Path to the base Qwen2.5-Omni model (for processor config)")
    parser.add_argument("--conversation_json", type=str, default=None,
                        help="Path to a JSON file containing the conversation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: cuda if available, else cpu)")
    parser.add_argument("--use_audio_in_video", action="store_true", default=True,
                        help="Whether to use audio track in video inputs")
    return parser.parse_args()


def load_model(ckpt_dir, base_ckpt, device, dtype):
    """Load the reward model and processor."""
    processor = OmniRewardProcessor.from_pretrained(base_ckpt)
    config = Qwen2_5OmniThinkerConfig.from_pretrained(base_ckpt)
    model = QwenOmniThinkerReward.from_pretrained(
        ckpt_dir, config=config, torch_dtype=dtype, device_map=device
    ).to(device)
    model.freeze_encoder()
    model.eval()
    return model, processor


def score_conversation(model, processor, conversation, use_audio_in_video=True):
    """Score a single conversation and return the reward logits."""
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=False, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=False,
        use_audio_in_video=use_audio_in_video,
        device="cpu",
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        output = model(**inputs, use_audio_in_video=use_audio_in_video)

    return output.logits


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from {args.ckpt_dir}...")
    model, processor = load_model(args.ckpt_dir, args.base_ckpt, device, dtype)
    print("Model loaded.")

    if args.conversation_json:
        with open(args.conversation_json, "r", encoding="utf-8") as f:
            conversation = json.load(f)
    else:
        # Minimal text-only example
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": "How are you today?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I'm doing great, thanks for asking!"}]},
        ]
        print("No conversation JSON provided. Using a minimal text-only example.")

    print("Scoring conversation...")
    logits = score_conversation(
        model, processor, conversation, use_audio_in_video=args.use_audio_in_video
    )
    print(f"Reward score: {logits.item():.4f}")


if __name__ == "__main__":
    main()
