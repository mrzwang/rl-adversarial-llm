#!/usr/bin/env python3
"""
Utility functions for training analysis and visualization.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import wandb

def save_rewards_to_csv(rewards, filename="all_rewards.csv"):
    """
    Save rewards to a CSV file.
    
    Args:
        rewards (list): List of reward values
        filename (str): Name of the CSV file to save to
    """
    df = pd.DataFrame(rewards, columns=["reward"])
    df.to_csv(filename, index=False)

def plot_rewards(rewards_file="all_rewards.csv", window=4):
    """
    Plot rewards from a CSV file.
    
    Args:
        rewards_file (str): Path to the CSV file containing rewards
        window (int): Window size for rolling mean
    """
    df = pd.read_csv(rewards_file, header=None, names=["reward"])
    
    # Plot 1: Raw rewards
    plt.figure()
    plt.plot(df["reward"])
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Raw Reward Values")
    plt.grid(True)
    plt.show()
    
    # Plot 2: Rolling mean for smoothing
    if len(df) >= window:
        rolling = df["reward"].rolling(window).mean()
        plt.figure()
        plt.plot(rolling)
        plt.xlabel("Iteration")
        plt.ylabel(f"Rolling Mean (window={window})")
        plt.title("Smoothed Reward Trend")
        plt.grid(True)
        plt.show()
    else:
        print(f"Not enough data points for rolling mean (need at least {window}, got {len(df)})")

def save_model_checkpoint(model, output_dir, name="model_checkpoint"):
    """
    Save a model checkpoint.
    
    Args:
        model: The model to save
        output_dir (str): Directory to save the model to
        name (str): Name of the checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, name))
    print(f"Model saved to {os.path.join(output_dir, name)}")

def zip_checkpoint(checkpoint_dir, output_zip):
    """
    Zip a checkpoint directory.
    
    Args:
        checkpoint_dir (str): Directory containing the checkpoint
        output_zip (str): Name of the output zip file
    """
    import subprocess
    subprocess.run(["zip", "-r", output_zip, checkpoint_dir])
    print(f"Checkpoint zipped to {output_zip}")

def setup_wandb(project_name="RL-Training"):
    """
    Set up Weights & Biases for experiment tracking.
    
    Args:
        project_name (str): Name of the W&B project
        
    Returns:
        wandb.run: The W&B run object
    """
    wandb.login()
    run = wandb.init(project=project_name)
    return run
