#!/usr/bin/env python3
"""
GRPO training script using length-based rewards.

Before running, install required packages:
    pip install datasets transformers trl peft accelerate bitsandbytes wandb
    pip install -U ninja packaging triton
    pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.6
"""

import torch
from trl import GRPOTrainer

# Import utility modules
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.grpo_models import load_model_and_tokenizer, apply_lora, create_grpo_config
from src.utils.data_utils import load_smoltldr_dataset
from src.utils.reward_utils import reward_len
from src.utils.training_utils import setup_wandb

def main():
    """Main training function"""
    # Setup wandb
    setup_wandb(project_name="GRPO")
    
    # Load dataset
    dataset = load_smoltldr_dataset()
    print(dataset)
    
    # Load model and tokenizer
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    model, tokenizer = load_model_and_tokenizer(
        model_id, 
        use_flash_attention=False  # Set to True if you have flash attention installed
    )
    
    # Apply LoRA
    model = apply_lora(
        model,
        r=16,
        lora_alpha=32,
        target_modules="all-linear"
    )
    print(model.print_trainable_parameters())

    # Create training arguments
    training_args = create_grpo_config(
        output_dir="GRPO",
        learning_rate=2e-5,
        batch_size=8,
        gradient_accumulation_steps=2,
        use_wandb=True
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[lambda completions, **kwargs: reward_len(completions, ideal_length=50, **kwargs)],
        args=training_args,
        train_dataset=dataset["train"],
    )
    
    # Train model
    trainer.train()
    
    # Merge and save model
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained("GRPO/merged_model")
    print("Model training complete and saved to GRPO/merged_model")


if __name__ == "__main__":
    main()
