#!/usr/bin/env python3
"""
GRPO model utilities for loading and configuring models.
"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

def load_model_and_tokenizer(model_id, use_flash_attention=False):
    """
    Load a model and tokenizer with the specified configuration.
    
    Args:
        model_id (str): The model ID to load from HuggingFace
        use_flash_attention (bool): Whether to use flash attention
        
    Returns:
        tuple: (model, tokenizer)
    """
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer

def apply_lora(model, r=16, lora_alpha=32, target_modules="all-linear"):
    """
    Apply LoRA to a model.
    
    Args:
        model: The model to apply LoRA to
        r (int): LoRA rank
        lora_alpha (int): LoRA alpha
        target_modules (str): Target modules for LoRA
        
    Returns:
        model: The model with LoRA applied
    """
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    return model

def create_grpo_config(output_dir="GRPO", learning_rate=2e-5, batch_size=8, 
                      gradient_accumulation_steps=2, use_wandb=True):
    """
    Create a GRPO training configuration.
    
    Args:
        output_dir (str): Directory to save outputs
        learning_rate (float): Learning rate
        batch_size (int): Batch size per device
        gradient_accumulation_steps (int): Gradient accumulation steps
        use_wandb (bool): Whether to use wandb for logging
        
    Returns:
        GRPOConfig: Training configuration
    """
    report_to = ["wandb"] if use_wandb else []
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=512,
        max_completion_length=96,
        num_generations=8,
        optim="adamw_8bit",
        num_train_epochs=1,
        bf16=False,  # Changed to False to disable BF16
        report_to=report_to,
        remove_unused_columns=False,
        logging_steps=1,
    )
    
    return training_args
