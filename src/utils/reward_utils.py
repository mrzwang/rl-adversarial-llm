#!/usr/bin/env python3
"""
Utility functions for reward calculation in reinforcement learning.
"""
import re
import io
import contextlib
import torch
from transformers import pipeline

def reward_len(completions, ideal_length=50, **kwargs):
    """
    Calculate rewards based on how close the completion length is to the ideal length.
    
    Args:
        completions (list): List of text completions
        ideal_length (int): Ideal length for completions
        
    Returns:
        list: List of rewards
    """
    return [-abs(ideal_length - len(completion)) for completion in completions]

def run_and_capture(code):
    """
    Executes code and captures its stdout output.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: Captured output or error message
    """
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {})  # empty global scope
    except Exception as e:
        return f"Execution error: {e}"
    return buffer.getvalue().strip()

def setup_prediction_model(model_name):
    """
    Set up a model for code prediction tasks.
    
    Args:
        model_name (str): Name of the model to use
        
    Returns:
        pipeline: Text generation pipeline
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto"
    )
    
    # Optionally: use GPU if available
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a text generation pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    return generator

def code_prediction_reward(completions, generator, **kwargs):
    """
    Calculate rewards based on code prediction accuracy.
    
    Args:
        completions (list): List of text completions
        generator: Text generation pipeline for prediction
        
    Returns:
        list: List of rewards
    """
    rewards = []
    
    for comp in completions:
        if not isinstance(comp, str):
            rewards.append(-1)  # invalid completion
            continue
        
        # Extract Code according to schema
        code = re.search(r"```python\s*\n(.*?)```", comp, re.DOTALL)
        if code:
            code = code.group(1).strip()
        else:
            code = ""
        
        expected_output = re.search(r"```output\s*(.*?)```", comp, re.DOTALL)
        if expected_output:
            expected_output = expected_output.group(1).strip()
        else:
            expected_output = ""
        
        prompt = (
            "Examine this code and predict the integer output.  \n"
            f"{code}\n\n"
            "Do not include any text, markdown, or explanation, just the number."
        )
        
        model_pred = generator(
            prompt, 
            max_new_tokens=200, 
            do_sample=True, 
            temperature=0.7
        )[0]['generated_text']
        
        model_pred = re.search(r"\b(-?\d+)\b", model_pred)  # search for first number
        model_pred = model_pred.group(1) if model_pred else ""
        
        try:
            model_pred = int(model_pred)
        except:
            model_pred = "ERROR: Conversion to integer failed"
        
        true = run_and_capture(code)
        
        print("---------------------------------------")
        print(f'model_pred: {model_pred}')
        print(f'code: {code}')
        print(f'expected output: {expected_output}')
        print(f'true output: {true}')
        print(f'original_completion: {comp}')
        print("---------------------------------------")
        
        reward = 1 if str(model_pred) == true else -1
        rewards.append(reward)
    
    return rewards
