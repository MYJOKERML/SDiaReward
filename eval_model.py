# eval_model.py - Evaluation script (supports checkpoint resume and sample-level score logging)
import torch
from transformers import Qwen2_5OmniThinkerConfig
from model.modeling_qwen_omni_thinker_reward import QwenOmniThinkerReward
from model.processing_qwen_omni_thinker_reward import OmniRewardProcessor
from qwen_omni_utils import process_mm_info
from datasets import load_from_disk
from collections import defaultdict
import json
from tqdm import tqdm
import traceback
import copy
import os
import argparse

# Command-line arguments
parser = argparse.ArgumentParser(description='Evaluate reward model')
parser.add_argument('--ckpt_dir', type=str, required=True,
                    help='Checkpoint directory to evaluate')
parser.add_argument('--base_ckpt', type=str, required=True,
                    help='Base model checkpoint for processor')
parser.add_argument('--dataset_path', type=str, required=True,
                    help='Dataset path')
parser.add_argument('--output_dir', type=str, default="./eval_outputs",
                    help='Output directory for results')
parser.add_argument('--resume', action='store_true',
                    help='Resume from checkpoint')
parser.add_argument('--random_init', action='store_true',
                    help='Use randomly initialized model instead of pretrained checkpoint')
args = parser.parse_args()

# Configuration
ckpt_dir = args.ckpt_dir
base_ckpt = args.base_ckpt
dataset_path = args.dataset_path
output_dir = args.output_dir

# Create output directory
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print("Loading processor and model...")
processor = OmniRewardProcessor.from_pretrained(base_ckpt)
config = Qwen2_5OmniThinkerConfig.from_pretrained(base_ckpt)

if args.random_init:
    print("Using randomly initialized model...")
    model = QwenOmniThinkerReward(config)
    model = model.to(dtype).to(device)
    ckpt_name = "random_init"
else:
    model = QwenOmniThinkerReward.from_pretrained(
        ckpt_dir, config=config, torch_dtype=dtype, device_map=device
    ).to(device)
    ckpt_name = os.path.basename(ckpt_dir.rstrip('/'))

model.freeze_encoder()
model.eval()

# Generate output filenames based on ckpt_name
sample_scores_file = os.path.join(output_dir, f"sample_scores_{ckpt_name}.jsonl")
checkpoint_file = os.path.join(output_dir, f"eval_checkpoint_{ckpt_name}.json")
results_file = os.path.join(output_dir, f"eval_results_{ckpt_name}.json")

print("Loading dataset...")
ds = load_from_disk(dataset_path)
val_ds = ds['validation']

USE_AUDIO_IN_VIDEO = True

# Checkpoint resume: load evaluated sample indices and scores
evaluated_indices = set()
sample_scores = []

def load_checkpoint():
    """Load checkpoint information"""
    global evaluated_indices, sample_scores

    if args.resume and os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, 'r') as f:
            ckpt_data = json.load(f)
        evaluated_indices = set(ckpt_data.get('evaluated_indices', []))
        print(f"Resuming from checkpoint: {len(evaluated_indices)} samples already evaluated")

    # Load saved sample scores
    if args.resume and os.path.exists(sample_scores_file):
        print(f"Loading sample scores from {sample_scores_file}...")
        with open(sample_scores_file, 'r') as f:
            for line in f:
                if line.strip():
                    sample_scores.append(json.loads(line))
        print(f"Loaded {len(sample_scores)} sample scores")

def save_checkpoint(current_idx):
    """Save checkpoint information"""
    ckpt_data = {
        'evaluated_indices': list(evaluated_indices),
        'current_idx': current_idx,
        'total_samples': len(val_ds)
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(ckpt_data, f)

def save_sample_score(sample_record):
    """Append a single sample score record"""
    with open(sample_scores_file, 'a') as f:
        f.write(json.dumps(sample_record, ensure_ascii=False) + '\n')
    sample_scores.append(sample_record)

def _clean_conversation(conv):
        """Clean None values from conversation content. Returns a deep copy."""
        conv = copy.deepcopy(conv)
        for turn in conv:
            if turn.get('content') and isinstance(turn['content'], list):
                turn['content'] = [
                    {k: v for k, v in item.items() if v is not None}
                    for item in turn['content']
                ]
        return conv

def get_score(conversation):
    """Get the score for a single conversation"""
    conversation = _clean_conversation(conversation)
    text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=False,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        device='cpu'
    )
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        score = model(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    return score.logits.item()

# Statistics
results = {
    'by_category': defaultdict(lambda: {'correct': 0, 'total': 0, 'chosen_scores': [], 'rejected_scores': []}),
    'by_source': defaultdict(lambda: {'correct': 0, 'total': 0, 'chosen_scores': [], 'rejected_scores': []}),
    'overall': {'correct': 0, 'total': 0, 'chosen_scores': [], 'rejected_scores': []}
}

# Load checkpoint
load_checkpoint()

# Rebuild statistics from saved sample scores
if sample_scores:
    print("Rebuilding statistics from saved sample scores...")
    for record in sample_scores:
        category = record.get('category', 'unknown')
        source = record.get('source', 'unknown')
        chosen_score = record['chosen_score']
        rejected_score = record['rejected_score']
        is_correct = record['is_correct']

        for key, val in [('by_category', category), ('by_source', source)]:
            results[key][val]['total'] += 1
            results[key][val]['chosen_scores'].append(chosen_score)
            results[key][val]['rejected_scores'].append(rejected_score)
            if is_correct:
                results[key][val]['correct'] += 1

        results['overall']['total'] += 1
        results['overall']['chosen_scores'].append(chosen_score)
        results['overall']['rejected_scores'].append(rejected_score)
        if is_correct:
            results['overall']['correct'] += 1

print(f"Starting evaluation on {len(val_ds)} samples...")
if evaluated_indices:
    print(f"Skipping {len(evaluated_indices)} already evaluated samples")

for idx in tqdm(range(len(val_ds))):
    # Checkpoint resume: skip already evaluated samples
    if idx in evaluated_indices:
        continue

    sample = val_ds[idx]

    chosen = sample['chosen']
    rejected = sample['rejected']
    category = sample.get('category', 'unknown')
    source = sample.get('source', 'unknown')

    if category is None:
        category = 'unknown'
    if source is None:
        source = 'unknown'

    try:
        chosen_score = get_score(chosen)
        rejected_score = get_score(rejected)
    except Exception as e:
        print(f"Error at sample {idx}: {e}")
        traceback.print_exc()
        error_record = {
            'idx': idx,
            'category': category,
            'source': source,
            'chosen_score': None,
            'rejected_score': None,
            'is_correct': None,
            'error': str(e)
        }
        save_sample_score(error_record)
        evaluated_indices.add(idx)
        save_checkpoint(idx)
        continue

    if chosen_score is None or rejected_score is None:
        print(f"Skipping sample {idx} due to error")
        continue

    # Correct if chosen score is higher than rejected
    is_correct = chosen_score > rejected_score

    # Record sample-level scores
    sample_record = {
        'idx': idx,
        'category': category,
        'source': source,
        'chosen_score': chosen_score,
        'rejected_score': rejected_score,
        'is_correct': is_correct,
        'score_margin': chosen_score - rejected_score
    }
    save_sample_score(sample_record)

    # Mark as evaluated
    evaluated_indices.add(idx)

    # Update statistics
    for key, val in [('by_category', category), ('by_source', source)]:
        results[key][val]['total'] += 1
        results[key][val]['chosen_scores'].append(chosen_score)
        results[key][val]['rejected_scores'].append(rejected_score)
        if is_correct:
            results[key][val]['correct'] += 1

    results['overall']['total'] += 1
    results['overall']['chosen_scores'].append(chosen_score)
    results['overall']['rejected_scores'].append(rejected_score)
    if is_correct:
        results['overall']['correct'] += 1

    # Save checkpoint and print progress every 100 samples
    if (idx + 1) % 100 == 0:
        save_checkpoint(idx)
        acc = results['overall']['correct'] / results['overall']['total'] * 100
        print(f"Progress: {idx+1}/{len(val_ds)}, Current accuracy: {acc:.2f}%")

# Final checkpoint save
save_checkpoint(len(val_ds) - 1)

# Print results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

print(f"\n{'='*60}")
print("Overall Results:")
print(f"{'='*60}")
total = results['overall']['total']
correct = results['overall']['correct']
acc = correct / total * 100 if total > 0 else 0
avg_chosen = sum(results['overall']['chosen_scores']) / len(results['overall']['chosen_scores']) if results['overall']['chosen_scores'] else 0
avg_rejected = sum(results['overall']['rejected_scores']) / len(results['overall']['rejected_scores']) if results['overall']['rejected_scores'] else 0
print(f"Total samples: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {acc:.2f}%")
print(f"Avg chosen score: {avg_chosen:.4f}")
print(f"Avg rejected score: {avg_rejected:.4f}")
print(f"Score margin: {avg_chosen - avg_rejected:.4f}")

print(f"\n{'='*60}")
print("Results by Category:")
print(f"{'='*60}")
for cat, stats in sorted(results['by_category'].items()):
    total = stats['total']
    correct = stats['correct']
    acc = correct / total * 100 if total > 0 else 0
    avg_chosen = sum(stats['chosen_scores']) / len(stats['chosen_scores']) if stats['chosen_scores'] else 0
    avg_rejected = sum(stats['rejected_scores']) / len(stats['rejected_scores']) if stats['rejected_scores'] else 0
    print(f"\nCategory: {cat}")
    print(f"  Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%")
    print(f"  Avg chosen score: {avg_chosen:.4f}")
    print(f"  Avg rejected score: {avg_rejected:.4f}")
    print(f"  Score margin: {avg_chosen - avg_rejected:.4f}")

print(f"\n{'='*60}")
print("Results by Source:")
print(f"{'='*60}")
for src, stats in sorted(results['by_source'].items()):
    total = stats['total']
    correct = stats['correct']
    acc = correct / total * 100 if total > 0 else 0
    avg_chosen = sum(stats['chosen_scores']) / len(stats['chosen_scores']) if stats['chosen_scores'] else 0
    avg_rejected = sum(stats['rejected_scores']) / len(stats['rejected_scores']) if stats['rejected_scores'] else 0
    print(f"\nSource: {src}")
    print(f"  Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%")
    print(f"  Avg chosen score: {avg_chosen:.4f}")
    print(f"  Avg rejected score: {avg_rejected:.4f}")
    print(f"  Score margin: {avg_chosen - avg_rejected:.4f}")

# Save detailed results to JSON
output_results = {
    'overall': {
        'total': results['overall']['total'],
        'correct': results['overall']['correct'],
        'accuracy': results['overall']['correct'] / results['overall']['total'] * 100 if results['overall']['total'] > 0 else 0,
        'avg_chosen_score': sum(results['overall']['chosen_scores']) / len(results['overall']['chosen_scores']) if results['overall']['chosen_scores'] else 0,
        'avg_rejected_score': sum(results['overall']['rejected_scores']) / len(results['overall']['rejected_scores']) if results['overall']['rejected_scores'] else 0,
    },
    'by_category': {},
    'by_source': {}
}

for cat, stats in results['by_category'].items():
    output_results['by_category'][cat] = {
        'total': stats['total'],
        'correct': stats['correct'],
        'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
        'avg_chosen_score': sum(stats['chosen_scores']) / len(stats['chosen_scores']) if stats['chosen_scores'] else 0,
        'avg_rejected_score': sum(stats['rejected_scores']) / len(stats['rejected_scores']) if stats['rejected_scores'] else 0,
    }

for src, stats in results['by_source'].items():
    output_results['by_source'][src] = {
        'total': stats['total'],
        'correct': stats['correct'],
        'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
        'avg_chosen_score': sum(stats['chosen_scores']) / len(stats['chosen_scores']) if stats['chosen_scores'] else 0,
        'avg_rejected_score': sum(stats['rejected_scores']) / len(stats['rejected_scores']) if stats['rejected_scores'] else 0,
    }

with open(results_file, 'w') as f:
    json.dump(output_results, f, indent=2, ensure_ascii=False)

print(f"\nDetailed results saved to {results_file}")
print(f"Sample-level scores saved to {sample_scores_file}")
print(f"Checkpoint file: {checkpoint_file}")
