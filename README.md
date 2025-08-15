# Reinforcement Learning for Language Models

This repository implements advanced reinforcement learning techniques for enhancing language model capabilities, featuring both PPO (Proximal Policy Optimization) and GRPO (General Reinforcement Policy Optimization) frameworks. The codebase focuses on RL fine-tuning of language models to improve code generation quality, consistency, and complexity—specifically targeting sub-4B parameter models to maximize their coding abilities.

A key idea behind this project is the adversarial training system where two language models compete against each other in real-time, continuously improving through reinforcement learning feedback loops. This approach creates a self-improving ecosystem where models learn to generate increasingly sophisticated code by attempting to outperform their opponents.

The modular architecture supports various model architectures, reward functions, and training configurations, making it adaptable for different code generation tasks and model sizes.

## Repository Structure

```
RL/
├── data/                    # Data storage directory
├── prerl/                   # Pre-reinforcement learning resources
├── rl/                      # Main training scripts
│   ├── Initial_RFT_Framework_GRPO.py  # GRPO implementation
│   ├── PPO_bothmodels.py              # PPO with competing models
│   └── RFT_single.py                  # Single model RFT implementation
├── src/                     # Source code modules
│   ├── models/              # Model implementations
│   │   ├── grpo_models.py   # GRPO model utilities
│   │   └── ppo_models.py    # PPO model utilities
│   └── utils/               # Utility functions
│       ├── data_utils.py    # Dataset loading and processing
│       ├── reward_utils.py  # Reward calculation functions
│       └── training_utils.py # Training utilities
└── convert_notebooks.py     # Utility to convert notebooks to Python scripts
```

## Setup

1. Install the required packages:
```bash
pip install datasets transformers trl peft accelerate bitsandbytes wandb
pip install git+https://github.com/huggingface/transformers
```

2. Set up your environment variables:
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

## Training Scripts

### GRPO Training
```bash
python rl/Initial_RFT_Framework_GRPO.py
```

### PPO with Competing Models
```bash
python rl/PPO_bothmodels.py
```

### Single Model RFT
```bash
python rl/RFT_single.py
```

## Models

The codebase supports various models from Hugging Face, including:
- Qwen/Qwen3-4B
- Qwen/Qwen2.5-Coder-0.5B-Instruct
- HuggingFaceTB/SmolLM-135M-Instruct

## Features

- Model loading and configuration with LoRA fine-tuning
- PPO and GRPO training implementations
- Reward calculation for various tasks
- Training utilities for checkpointing and visualization
- Dataset loading and processing

## License

[MIT License](LICENSE)
