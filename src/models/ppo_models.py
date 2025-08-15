#!/usr/bin/env python3
"""
PPO model utilities for loading and configuring models.
"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    BitsAndBytesConfig
)
from trl import PPOConfig, AutoModelForCausalLMWithValueHead

def load_model_and_tokenizer(model_id, use_8bit=False, use_4bit=False):
    """
    Load a model and tokenizer with the specified configuration.
    
    Args:
        model_id (str): The model ID to load from HuggingFace
        use_8bit (bool): Whether to use 8-bit quantization
        use_4bit (bool): Whether to use 4-bit quantization
        
    Returns:
        tuple: (model, tokenizer)
    """
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    
    if use_8bit:
        model_kwargs["load_in_8bit"] = True
    elif use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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

def add_value_head(model):
    """
    Add a value head to a model for PPO training.
    
    Args:
        model: The model to add a value head to
        
    Returns:
        model: The model with a value head
    """
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    return model

def create_ppo_config(output_dir="PPO", learning_rate=2e-5, batch_size=8, 
                     gradient_accumulation_steps=2, use_wandb=True):
    """
    Create a PPO training configuration.
    
    Args:
        output_dir (str): Directory to save outputs
        learning_rate (float): Learning rate
        batch_size (int): Batch size per device
        gradient_accumulation_steps (int): Gradient accumulation steps
        use_wandb (bool): Whether to use wandb for logging
        
    Returns:
        PPOConfig: Training configuration
    """
    report_to = ["wandb"] if use_wandb else []
    
    training_args = PPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=512,
        optim="adamw_8bit",
        num_train_epochs=1,
        bf16=False,
        report_to=report_to,
        remove_unused_columns=False,
        logging_steps=1,
    )
    
    return training_args

def ppo_train_one_step(batch, ppo_trainer, reward_fn, generation_kwargs, 
                      save_path=None, device="cuda", store_codes=True):
    """
    Run one step of PPO training.
    
    Args:
        batch: Batch of data
        ppo_trainer: PPO trainer
        reward_fn: Reward function
        generation_kwargs: Generation kwargs
        save_path: Path to save outputs
        device: Device to use
        store_codes: Whether to store codes
        
    Returns:
        stats: Training statistics
    """
    query_tensors = batch["input_ids"].to(device)
    query_list = list(query_tensors.unbind(dim=0))

    for i, q in enumerate(query_list):
        # 1D tensor?
        assert q.ndim == 1, f"query_list[{i}] has shape {q.shape}"
        # no negative IDs
        assert int(q.min()) >= 0, f"Negative token in example {i}"
        # below vocab
        assert int(q.max()) < ppo_trainer.model.config.vocab_size, (
            f"Token id {int(q.max())} >= vocab size "
            f"{ppo_trainer.model.config.vocab_size} at example {i}"
        )

    # Generate responses
    response_tensors = ppo_trainer.generate(
        query_list,
        return_prompt=False,
        generate_ref_response=False,
        **generation_kwargs,
    )

    response_tensors = [r.to(device) for r in response_tensors]
    
    # Compute rewards
    texts = [ppo_trainer.tokenizer.decode(r.squeeze()) for r in response_tensors]
    rewards = reward_fn(texts)
    rewards = [torch.tensor(reward, device=device) for reward in rewards]
    
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    return stats
