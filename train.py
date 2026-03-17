import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
import torch.multiprocessing as mp

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig, Qwen2_5OmniForConditionalGeneration, HfArgumentParser
from model.modeling_qwen_omni_thinker_reward import QwenOmniThinkerReward
from model.processing_qwen_omni_thinker_reward import OmniRewardProcessor
from qwen_omni_utils import process_mm_info
from datasets import load_dataset, load_from_disk
import warnings
from trl import (
    ModelConfig,
    RewardConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
from peft import LoraConfig, TaskType
from trainer.multimodal_reward_trainer import MultimodalRewardTrainer


from utils.load_utils import load_dataset_clean
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See
            https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    dataset_type: str = field(default="datasets", metadata={"help": "Dataset type: 'datasets' or 'json'"})
    train_json: str = field(default=None, metadata={"help": "Path to the training JSON file."})
    val_json: str = field(default=None, metadata={"help": "Path to the validation JSON file."})
    test_json: str = field(default=None, metadata={"help": "Path to the test JSON file."})
    dataset_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode."},
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    print("Report to:", training_args.report_to)
    print("Logging dir:", training_args.logging_dir)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    if training_args.bf16:
        print('Using bfloat16 precision')
        torch_dtype = torch.bfloat16
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    processor = OmniRewardProcessor.from_pretrained(
       model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    config = Qwen2_5OmniThinkerConfig.from_pretrained(model_args.model_name_or_path)
    model = QwenOmniThinkerReward.from_pretrained(
        model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    # Freeze audio and visual encoders
    print("\nFreezing audio and visual encoders...")
    # When using LoRA, only freeze encoders (LoRA handles the LLM backbone)
    # Without LoRA, freeze encoders and LLM backbone, only train pooler and score_head
    model.freeze_encoder(freeze_text=not model_args.use_peft)

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen parameters: {frozen_count}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}\n")

    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = config.text_config.pad_token_id
    print("Padding token ID:", model.config.pad_token_id)

    # If post-training a base model, use ChatML as the default template
    if processor.chat_template is None:
        model, processor = setup_chat_format(model, processor)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    ##############
    # Load dataset
    ##############
    print("Loading dataset...")
    if script_args.dataset_type == 'datasets':
        dataset = load_from_disk(script_args.dataset_name)
    elif script_args.dataset_type == 'json':
        dataset = load_dataset('json', data_files={'train': script_args.train_json, 'val': script_args.val_json, 'test': script_args.test_json})
    else:
        raise ValueError(f"Unsupported dataset_type: {script_args.dataset_type}. Use 'datasets' or 'json'.")
    print("Dataset loaded successfully.")

    ##########
    # Training
    ##########
    print("Initializing RewardTrainer...")

    peft_config = None
    if model_args.use_peft:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            task_type=TaskType.SEQ_CLS,
            modules_to_save=["attn_pool", "score_head"],
        )
        print(f"Using LoRA with modules_to_save: {peft_config.modules_to_save}")

    trainer = MultimodalRewardTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
    )
    print('Trainer initialized.')

    if training_args.resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        print("Starting training from scratch...")
        trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
