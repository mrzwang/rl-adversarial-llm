#!/usr/bin/env python3
"""
RFT (Reinforcement Fine-Tuning) script using code prediction rewards.

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
from src.utils.data_utils import create_prompt_dataset, get_code_generation_prompt
from src.utils.reward_utils import code_prediction_reward, setup_prediction_model
from src.utils.training_utils import setup_wandb


def main():
    """Main training function"""
    # Setup wandb
    setup_wandb(project_name="GRPO-Code-Prediction")
    
    # Get prompt and create dataset
    prompt = get_code_generation_prompt()
    dataset = create_prompt_dataset(prompt, size=1000)
    print(dataset)

    # Load model and tokenizer for training
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

    # Set up prediction model for reward calculation
    prediction_model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    generator = setup_prediction_model(prediction_model_name)


    # Create training arguments
    training_args = create_grpo_config(
        output_dir="GRPO-Code",
        learning_rate=2e-5,
        batch_size=8,
        gradient_accumulation_steps=2,
        use_wandb=True
    )
    
    # Create trainer with reward function
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[lambda completions, **kwargs: code_prediction_reward(completions, generator, **kwargs)],
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train model
    trainer.train()
    
    # Merge and save model
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained("GRPO-Code/merged_model")
    print("Model training complete and saved to GRPO-Code/merged_model")


if __name__ == "__main__":
    main()
