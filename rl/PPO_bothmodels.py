#!/usr/bin/env python3
"""
PPO training script for two competing models.

Before running, install required packages:
    pip install datasets transformers trl peft accelerate bitsandbytes wandb
    pip install git+https://github.com/huggingface/transformers
"""

import torch
import os
import sys
import re
import json
import random
import datetime
import itertools
from tqdm import tqdm

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Cell 4
from huggingface_hub import login
from google.colab import userdata
import os, re, tempfile, subprocess, textwrap, random, json, datetime
import torch, itertools
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    BitsAndBytesConfig, set_seed, DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
#from trl import GRPOTrainer, GRPOConfig
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead
from tqdm import tqdm

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Import utility modules
from src.models.ppo_models import load_model_and_tokenizer, apply_lora, add_value_head, create_ppo_config, ppo_train_one_step
from src.utils.training_utils import save_rewards_to_csv, plot_rewards, save_model_checkpoint, zip_checkpoint, setup_wandb
from src.utils.reward_utils import execute_code

PLAYER_MODEL_ID = "Qwen/Qwen3-4B"
#PLAYER_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE          = "cpu"
GLOBAL_SEED     = 42
MAX_NEW_TOKENS  = 512
GEN_BATCH       = 8
BATCH_SIZE      = 2
set_seed(GLOBAL_SEED)
torch.backends.cuda.matmul.allow_tf32 = True

HASH_FENCE_RE  = re.compile(r"(?m)^\s*#{3}\s*$")
BTICK_FENCE_RE = re.compile(r"(?:python)?\n([\s\S]*?)", re.I)
PRED_RE        = re.compile(r"<\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*>")

def extract_code_and_pred(txt: str):
    """
    From a full LLM completion, return (code_block:str, numeric_prediction:float).
    Raises ValueError if either chunk is missing.
    """
    txt = txt.replace("python", "")
    code, search_from = None, None

    fences = list(HASH_FENCE_RE.finditer(txt))
    if len(fences) >= 2:                         # ### … ###
        code  = txt[fences[0].end():fences[1].start()].strip()
        search_from = fences[1].end()
    else:                                        # ```python … ```
        m = BTICK_FENCE_RE.search(txt)
        if not m:
            raise ValueError("could not find fenced code")
        code        = m.group(1).strip()
        search_from = m.end()

    m_pred = PRED_RE.search(txt, pos=search_from)
    if not m_pred:
        raise ValueError("no numeric prediction found after code block")
    return code, float(m_pred.group(1))

def run_python(code: str, timeout_s=5) -> float:
    """Run Python code and return the last numeric output."""
    result = execute_code(code, timeout_seconds=timeout_s)
    if result['status'] != 'success':
        raise RuntimeError(result['error'])
    
    # Extract the last numeric value from stdout
    for line in reversed(result['stdout'].strip().splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    raise RuntimeError("no numeric output")

# Load tokenizer and model using utility function
player_tokenizer, player_base_model = load_model_and_tokenizer(
    model_id=PLAYER_MODEL_ID,
    device=DEVICE,
    padding_side="left",
    enable_thinking=False
)


# Create LoRA configuration
lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM"
)


# Create PPO config using utility function
player_cfg = create_ppo_config(
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=2,
    seed=GLOBAL_SEED,
    steps=1
    #logging_steps=10,
    #save_steps=50,
    #max_steps=10,
)

opp_cfg = create_ppo_config(
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=2,
    seed=GLOBAL_SEED,
    steps=1,
    #logging_steps=10,
    #save_steps=50,
    #max_steps=10,
)

# Model loading is now handled by utility functions

@torch.no_grad()
def predict_output(model, code: str) -> float | None:
    prompt = (
        "I will show you a Python script.\n"
        "Predict the numeric value it prints *exactly*.\n\n"
        f"python\n{code}\n"
    )
    toks = player_tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out  = model.generate(**toks, max_new_tokens=32, temperature=0.0,
                          do_sample=False, eos_token_id=player_tokenizer.eos_token_id)
    txt  = player_tokenizer.decode(out[0], skip_special_tokens=True)
    m    = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", txt)
    return float(m.group()) if m else None
def player_reward(prompts, completions, **_):
    """+1 if Player right & Opponent wrong, -1 vice versa, 0 otherwise."""
    rews = []
    for comp in completions:
        try:
            code, p_pred = extract_code_and_pred(comp)
            true_out     = run_python(code)
            COLLECTED_CODES.append(code)

        except Exception:
            rews.append(-1.0);
            COLLECTED_CODES.append('NA')
            continue
        o_pred   = predict_output(opp_model, code)
        p_ok     = abs(p_pred-true_out)  < 1e-6
        o_ok     = o_pred is not None and abs(o_pred-true_out) < 1e-6
        rews.append(+1.0 if  p_ok and not o_ok else
                    -1.0 if not p_ok and  o_ok else 0.0)

        print("reward", rews)
        with open("all_rewards.csv","a") as f:
          for r in rews:
              f.write(f"{r}\n")

    return rews

def opp_reward(prompts, completions, **_):
    """Mirror image for the Opponent."""
    rews = []
    print(completions)
    for comp in completions:
        try:
            code, o_pred = extract_code_and_pred(comp)
            true_out     = run_python(code)
            print(code)
            print(o_pred)
        except Exception:
            rews.append(-1.0); continue
        p_pred  = predict_output(player_model, code)
        o_ok    = abs(o_pred-true_out)  < 1e-6
        p_ok    = p_pred is not None and abs(p_pred-true_out) < 1e-6
        rews.append(+1.0 if  o_ok and not p_ok else
                    -1.0 if not o_ok and  p_ok else 0.0)

        print("reward", rews)
        with open("all_rewards.csv","a") as f:
          for r in rews:
              f.write(f"{r}\n")
    return rews

BASE_PROMPT = textwrap.dedent("""\
  We're going to play a game to test your capabilities.
  1. Write a stand‑alone python script between ### … ### that prints one numeric value.
  2. Predict that numeric value enclosed in <>.
  3. You score +1 if you are right and the Opponent is wrong; 0 otherwise.
  FORMAT:
  ###
  code
  ###
  <prediction>
""")
RAW_DS = Dataset.from_dict({"prompt": [BASE_PROMPT]*4_000})

def tokenize(ds):
    return player_tokenizer(ds["prompt"], truncation=True, max_length=512)


def create_batched_dataset(dataset, tokenizer, collator, batch_size=8):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,  # or your custom collator
    )

    batched_data = []
    for batch in dataloader:
        # Convert all tensors to device-agnostic (will move to cuda later)
        batched_data.append({k: v for k, v in batch.items()})

    return batched_data  # list of dicts, one dict per batch

train_ds      = RAW_DS.map(tokenize, batched=True, remove_columns=["prompt"])
train_ds.set_format("torch", columns=["input_ids", "attention_mask"])
collator      = DataCollatorWithPadding(player_tokenizer, return_tensors="pt")

batched_train_ds = create_batched_dataset(train_ds, tokenizer=player_tokenizer, batch_size=BATCH_SIZE, collator=collator)

if player_tokenizer.pad_token_id is None:
    player_tokenizer.pad_token = player_tokenizer.eos_token
    player_model.config.pad_token_id = player_tokenizer.pad_token_id

# Cell 6

def ppo_train_one_step(
    batch,
    ppo_trainer,
    reward_fn,
    generation_kwargs,
    save_path=None,
    device="cuda",
    store_codes=True,
):
    #batch = ppo_trainer.dataset[batch_id]
    #print(ppo_trainer.dataset)
    #print(batch)
    query_tensors = batch["input_ids"].to(device)

    query_list = list(query_tensors.unbind(dim=0))            # List of B tensors, each [L]

    for i, q in enumerate(query_list):
      # 1D tensor?
      assert q.ndim == 1, f"query_list[{i}] has shape {q.shape}"
      # no negative IDs
      assert int(q.min()) >= 0,     f"Negative token in example {i}"
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


    batch['response'] = player_tokenizer.batch_decode(response_tensors)
    response_texts = batch['response']
    prompt_texts = player_tokenizer.batch_decode(query_tensors)

    #prompt_texts = player_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    rewards = reward_fn(prompt_texts, response_texts)
    print(rewards)
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float)

    rewards = rewards.to(device)
    reward_list = [r.unsqueeze(0) for r in rewards.unbind(dim=0)]

    stats = ppo_trainer.step(query_list, response_tensors, reward_list)
    ppo_trainer.log_stats(stats, batch, reward_list)

    if save_path:
        ppo_trainer.save_pretrained(save_path)

    return stats

# Cell 7
# Cell 8
print("Vocab size:", player_model.config.vocab_size)

# Cell 9

player_cfg.output_dir = "./player_ckpt"
opp_cfg.output_dir = "./opp_ckpt"

# Apply LoRA and add value head to player model using utility functions
player_model = apply_lora(player_base_model, lora_cfg)
player_model = add_value_head(player_model)

# Create reference model for player
_, player_ref = load_model_and_tokenizer(
    model_id=PLAYER_MODEL_ID,
    device=DEVICE,
    load_in_8bit=False,
    torch_dtype=torch.float16
)

# Load and prepare opponent model
_, opp_base_model = load_model_and_tokenizer(
    model_id=PLAYER_MODEL_ID,
    device=DEVICE,
    load_in_8bit=True,
    torch_dtype=torch.float16
)

# Apply LoRA and add value head to opponent model
opp_model = apply_lora(opp_base_model, lora_cfg)
opp_model = add_value_head(opp_model)

# Create reference model for opponent
_, opp_ref = load_model_and_tokenizer(
    model_id=PLAYER_MODEL_ID,
    device=DEVICE,
    load_in_8bit=False,
    torch_dtype=torch.float16
)

# Create PPO trainer for player
player_trainer = PPOTrainer(
    config=player_cfg,
    model=player_model,
    ref_model=player_ref,
    tokenizer=player_tokenizer,
    dataset=train_ds,
    data_collator=collator,
    #reward_model=dummy_reward_model,
    #value_model = player_val,
    )

# Create PPO trainer for opponent
opp_trainer = PPOTrainer(
    config=opp_cfg,
    model=opp_model,
    ref_model=opp_ref,
    tokenizer=player_tokenizer,  # Using the same tokenizer for both models
    dataset=train_ds,
    data_collator=collator,
    #value_model=opp_model.base_model,
)

generation_kwargs_player = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": 0.8,
    "do_sample": True,
    "pad_token_id": player_tokenizer.pad_token_id,
    "use_cache": False,
}
generation_kwargs_opp = {
    "max_new_tokens": 32,
    "temperature": 0.0,
    "do_sample": False,
    "pad_token_id": player_tokenizer.pad_token_id,
    "use_cache": False,
}

batch_iter = iter(player_trainer.dataloader)

NUM_ROUNDS = 15
for round_idx in range(NUM_ROUNDS):


    COLLECTED_CODES = []
    print(f"Round {round_idx+1}/{NUM_ROUNDS}: Training Player")
    for step in range(player_cfg.steps):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(player_trainer.dataloader)
            batch = next(batch_iter)

        stats = ppo_train_one_step(
            batch=batch,
            ppo_trainer=player_trainer,
            reward_fn=player_reward,
            generation_kwargs=generation_kwargs_player,
            save_path=player_cfg.output_dir,
            device=DEVICE,
        )
    print(f"Player step {step+1}")# stats:", stats)

    codes = COLLECTED_CODES
    print(f"\nPlayer RL produced {len(COLLECTED_CODES)} code.")
    print(codes)
    # Build Opponent dataset
    def make_opp_prompt(code: str) -> str:
        return (
            "Here’s the code – predict its output:\n\n"
            f"python\n{code}\n"
        )

    def opp_prompt(code):
        return textwrap.dedent("""\
                              Here is a block of code – I want you to predict its output:\n\npython\n+{code}+\n.
                              1. Write the code between ### … ###.
                              2. Predict the numeric output of the code and enclose it in theses brackets: <>.
                              3. You score +1 if you are right; 0 otherwise.
                              Example format:
                              ###
                              print(1 + 2)
                              ###
                              <3>
                              """).format(code=code)
        # return  '''\
        #         Here is a block of code – predict its output:\n\npython\n+{code}+\n.
        #         Output your response using the following output format:
        #         ###
        #         {code}
        #         ###
        #         <output>
        #         '''.format(code=code)

    opp_ds_raw = Dataset.from_dict({"prompt":[opp_prompt(c) for c in codes]})
    opp_ds      = opp_ds_raw.map(tokenize, batched=True, remove_columns=["prompt"])
    print(f"Round {round_idx+1}/{NUM_ROUNDS}: Training Opponent ===")
    opp_trainer = PPOTrainer(
        config=opp_cfg,
        model=opp_model,
        ref_model=opp_ref,
        tokenizer=player_tokenizer,  # Using the same tokenizer for both models
        dataset=opp_ds,  # You'll generate this each round
        data_collator=collator,

    )
    opp_iter = iter(opp_trainer.dataloader)

    for step in range(opp_cfg.steps):
        try:
            batch = next(opp_iter)
        except StopIteration:
            # restart if we run out of examples
            opp_iter = iter(opp_trainer.dataloader)
            batch    = next(opp_iter)

        stats = ppo_train_one_step(
            batch=batch,
            ppo_trainer=opp_trainer,
            reward_fn=opp_reward,
            generation_kwargs=generation_kwargs_opp,
            save_path=opp_cfg.output_dir,
            device=DEVICE,
        )

print("\nTraining complete.")
print(" Player weights saved to:", player_cfg.output_dir)
print(" Opponent weights saved to:", opp_cfg.output_dir)

# Cell 10
import pandas as pd
import matplotlib.pyplot as plt

# Load reward history from disk
df = pd.read_csv("all_rewards.csv", header=None, names=["reward"])

# Plot 1: Raw reward per iteration
plt.figure()
plt.plot(df["reward"].values)
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Reward per Iteration")
plt.grid(True)
plt.show()

# Plot 2: Rolling mean for smoothing
window = 4
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

# Cell 11
# Zip the checkpoint using utility function
zip_checkpoint("player_ckpt", "player-rl.zip")

# Cell 12
# Zip the opponent checkpoint using utility function
zip_checkpoint("opp_ckpt", "opp-rl.zip")
