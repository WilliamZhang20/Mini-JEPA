#!/usr/bin/env python
"""Count parameters in JEPA model checkpoints."""

import torch
from pathlib import Path
from collections import defaultdict
import json


def count_parameters(model_state_dict):
    """Count total and trainable parameters in a model state dict."""
    total = 0
    trainable = 0
    for name, param in model_state_dict.items():
        num_params = param.numel()
        total += num_params
        # Assume all params in saved state_dict are trainable
        trainable += num_params
    return total, trainable


def format_params(num):
    """Format parameter count with K/M suffix."""
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num}"


def main():
    runs_dir = Path("/u5/w223zhan/jepa-mini/runs")
    
    # Find all model files
    model_files = sorted(runs_dir.glob("**/checkpoints/*_model.pt"))
    
    if not model_files:
        print("No model files found!")
        return
    
    results = {}
    
    for model_path in model_files:
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict = checkpoint.get("model", checkpoint)
            
            total, trainable = count_parameters(model_state_dict)
            
            # Extract task and variant from path
            parts = model_path.parts
            task = parts[-3]  # fetch_reach, fetch_push, fetch_pick_place, or dry
            variant = model_path.stem.replace("_model", "")
            
            key = f"{task}/{variant}"
            results[key] = {
                "path": str(model_path),
                "total_params": total,
                "formatted": format_params(total)
            }
            
            print(f"{key:60s} | {format_params(total):>10s} params ({total:>12,d})")
        
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    print("\n" + "="*80)
    
    # Group by task and find latest
    by_task = defaultdict(list)
    for key, data in results.items():
        task = key.split("/")[0]
        by_task[task].append((key, data))
    
    print("\nLatest/Production Models:")
    print("-" * 80)
    
    production_models = [
        "fetch_reach/reach_goal_focus_deadline_model",
        "fetch_reach/goal_focus_slurm_1447750_model",
        "fetch_pick_place/pickplace_v2_model",
        "fetch_push/push_v2_model",
    ]
    
    total_prod_params = 0
    for model_name in production_models:
        full_key = model_name.replace("_model", "")
        if full_key in results:
            data = results[full_key]
            print(f"{full_key:60s} | {data['formatted']:>10s}")
            total_prod_params += data['total_params']
    
    print("-" * 80)
    print(f"{'Total production models':60s} | {format_params(total_prod_params):>10s}")


if __name__ == "__main__":
    main()
