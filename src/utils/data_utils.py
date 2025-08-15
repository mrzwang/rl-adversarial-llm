#!/usr/bin/env python3
"""
Utility functions for data preparation and handling.
"""
from datasets import load_dataset, Dataset

def load_smoltldr_dataset():
    """
    Load the smoltldr dataset.
    
    Returns:
        dataset: The loaded dataset
    """
    dataset = load_dataset("mlabonne/smoltldr")
    return dataset

def create_prompt_dataset(prompt, size=1000):
    """
    Create a dataset with repeated prompts.
    
    Args:
        prompt (str): The prompt to repeat
        size (int): Number of times to repeat the prompt
        
    Returns:
        Dataset: Dataset with repeated prompts
    """
    dataset = Dataset.from_dict({
        "prompt": [prompt] * size
    })
    return dataset

def get_code_generation_prompt():
    """
    Get a prompt for code generation tasks.
    
    Returns:
        str: Formatted prompt
    """
    prompt = (
        "Write a Python program that is difficult for another model trained on SmollM-135M-Instruct data to predict. "
        "The program should return a singular integer value. "
        "Then, show only the exact output of running your program.\n\n"

        "Format your response exactly like these examples:\n\n"

        "```python\n"
        "def tricky():\n"
        "    return int('0b1011', 2)\n"
        "print(tricky())\n"
        "```\n"
        "```output\n"
        "11\n"
        "```\n\n"

        "```python\n"
        "def f():\n"
        "    return sum([i % 3 for i in range(10)])\n"
        "print(f())\n"
        "```\n"
        "```output\n"
        "10\n"
        "```\n\n"

        "Now you try:\n"
    )
    return prompt
