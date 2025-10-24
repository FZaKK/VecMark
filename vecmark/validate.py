"""One-Class Validation Script (New Version)
- Supports oneclass_model.pt (with threshold) or old checkpoint (fallback to base_threshold)
- Uses same mean/std normalization as training
- Uses consistency error score (MSE between predicted & modulated noise) as metric
- Multiple Monte Carlo sampling with CSV report output
"""

import argparse
import os
import sys
import time
import csv
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

from model import DynamicDiffusion, UNet1D

class SimpleVectorDataset(Dataset):
    """Dataset wrapper for vector data files (.npy, .pkl)"""
    
    def __init__(self, path: str):
        data = self._load_any(path)
        
        if isinstance(data, np.ndarray):
            self.vectors = data if data.ndim == 2 else data.reshape(-1, data.shape[-1])
        elif isinstance(data, list):
            try:
                array_data = np.array(data)
                if array_data.ndim == 2:
                    self.vectors = array_data
                else:
                    self.vectors = array_data.reshape(-1, 1)
            except (ValueError, TypeError):
                processed_data = []
                for item in data:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        processed_data.append(np.array(item).flatten())
                    else:
                        processed_data.append([float(item)])
                self.vectors = np.array(processed_data)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                    if isinstance(value[0], (int, float, np.number)):
                        self.vectors = np.array(value)
                        if self.vectors.ndim == 1:
                            self.vectors = self.vectors.reshape(-1, 1)
                        break
            else:
                raise ValueError("Cannot extract valid numerical data from dictionary")
        else:
            raise TypeError(f"Unsupported data type {type(data)}")
            
        if self.vectors.ndim != 2:
            self.vectors = self.vectors.reshape(1, -1)
            
        self.length = self.vectors.shape[0]
        print(f"Loaded {self.length} samples, each with dimension {self.vectors.shape[1]}")

    def _load_any(self, path):
        """Load data from .npy or .pkl file"""
        if path.endswith(".npy"):
            return np.load(path, allow_pickle=True)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file format")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(self.vectors[idx], dtype=torch.float32)

@torch.no_grad()
def anomaly_score_batch(model, diffusion, x, mc_samples=8, mean=None, std=None):
    """Calculate consistency error score for batch of samples with forced normalization"""
    if mean is not None and std is not None:
        x = (x - mean) / torch.clamp(std, min=1e-6)

    B = x.size(0)
    scores = torch.zeros(B, device=x.device)
    for _ in range(mc_samples):
        t = torch.randint(1, diffusion.noise_steps, (B,), device=x.device)
        x_t, mod_noise = diffusion.noise_images(x, t)
        pred_noise = model(x_t, t)
        mse = torch.mean((pred_noise - mod_noise) ** 2, dim=(1, 2))
        scores += mse
    return scores / mc_samples

@torch.no_grad()
def load_model_package(path, device):
    """Load model package with configuration and weights"""
    ckpt = torch.load(path, map_location=device)
    stats = ckpt.get("stats", {})
    
    config = ckpt.get("config", {})
    
    vector_len = config.get('vector_len', ckpt.get('vector_len', stats.get('vector_len')))
    if vector_len is None:
        raise KeyError("vector_len missing in checkpoint")

    base_channels = config.get('base_channels', ckpt.get('base_channels', 128))
    num_layers = config.get('num_layers', ckpt.get('num_layers', 6))
    modulator_hidden = config.get('modulator_hidden', ckpt.get('modulator_hidden', 128))
    noise_steps = config.get('noise_steps', ckpt.get('noise_steps', stats.get('noise_steps', 1000)))
    beta_start = config.get('beta_start', ckpt.get('beta_start', 1e-4))
    beta_end = config.get('beta_end', ckpt.get('beta_end', 0.02))
    
    print(f"Loading model with configuration:")
    print(f"  vector_len: {vector_len}")
    print(f"  base_channels: {base_channels}")
    print(f"  num_layers: {num_layers}")
    print(f"  modulator_hidden: {modulator_hidden}")
    print(f"  noise_steps: {noise_steps}")
    print(f"  beta_start: {beta_start}")
    print(f"  beta_end: {beta_end}")

    model = UNet1D(
        vector_len=vector_len, 
        base_channels=base_channels, 
        num_layers=num_layers
    ).to(device)
    
    diffusion = DynamicDiffusion(
        noise_steps=noise_steps, 
        beta_start=beta_start,
        beta_end=beta_end,
        vector_len=vector_len, 
        device=device,
        modulator_hidden=modulator_hidden
    )
    
    model.load_state_dict(ckpt["model_state"])
    diffusion.noise_modulator.load_state_dict(ckpt["modulator_state"])

    threshold = ckpt.get("threshold", None)
    return model.eval(), diffusion, stats, threshold

def main(args):
    """Main validation workflow"""
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, diffusion, stats, pkg_thr = load_model_package(args.model_path, device)

    mean = stats.get("mean", None)
    std = stats.get("std", None)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1,1,-1) if mean is not None else None
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1,1,-1) if std is not None else None

    if pkg_thr is not None:
        threshold = float(pkg_thr)
        thr_src = "package.threshold"
    else:
        threshold = float(stats.get("base_threshold", 0.1))
        thr_src = "base_threshold"
    
    print(f"Using threshold: {threshold:.6f} (source: {thr_src})")

    ds = SimpleVectorDataset(args.vector_path)
    if args.limit > 0:
        ds.vectors = ds.vectors[:args.limit]
        ds.length = ds.vectors.shape[0]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded dataset: {len(ds)} samples from {args.vector_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"validate_{ts}.csv")
    
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "score"])

    idx = 0
    all_scores = []
    start_time = time.time()
    
    for batch in tqdm(dl, desc="Validating"):
        x = batch.to(device).unsqueeze(1)
        scores = anomaly_score_batch(model, diffusion, x, 
                                    mc_samples=args.mc_samples,
                                    mean=mean_t, 
                                    std=std_t)
        
        rows_to_write = []
        for s in scores.tolist():
            rows_to_write.append([idx, f"{s:.6f}"])
            all_scores.append(s)
            idx += 1
        
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerows(rows_to_write)
    
    elapsed = time.time() - start_time
    scores_np = np.array(all_scores)
    preds = ["OTHER" if s >= threshold else "A" for s in all_scores]
    n_total = len(preds)
    n_a = preds.count("A")
    n_other = preds.count("OTHER")
    
    print(f"\nValidation completed in {elapsed:.2f} seconds")
    print(f"Results saved to {csv_path}")
    print(f"Score statistics: min={scores_np.min():.6f}, max={scores_np.max():.6f}")
    print(f"                 mean={scores_np.mean():.6f}, std={scores_np.std():.6f}")
    print(f"\nTotal samples: {n_total}")
    print(f"Predicted A: {n_a} ({n_a/n_total:.2%})")
    print(f"Predicted OTHER: {n_other} ({n_other/n_total:.2%})")
    
    stats_path = os.path.join(args.out_dir, f"validation_stats_{ts}.txt")
    with open(stats_path, "w") as f:
        f.write("VALIDATION STATISTICS\n")
        f.write("====================\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.vector_path}\n")
        f.write(f"Threshold: {threshold:.6f} (source: {thr_src})\n")
        f.write(f"MC Samples: {args.mc_samples}\n")
        f.write(f"Total samples: {n_total}\n")
        f.write(f"Predicted A: {n_a} ({n_a/n_total:.2%})\n")
        f.write(f"Predicted OTHER: {n_other} ({n_other/n_total:.2%})\n")
        f.write(f"Score min: {scores_np.min():.6f}\n")
        f.write(f"Score max: {scores_np.max():.6f}\n")
        f.write(f"Score mean: {scores_np.mean():.6f}\n")
        f.write(f"Score std: {scores_np.std():.6f}\n")
        f.write(f"Elapsed time: {elapsed:.2f} seconds\n")
        f.write(f"Samples per second: {n_total/elapsed:.2f}\n")
    
    print(f"Detailed statistics saved to {stats_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to model package (.pt file)")
    p.add_argument("--vector_path", type=str, required=True, help="Path to input vectors (.npy or .pkl file)")
    p.add_argument("--out_dir", type=str, default="./validation_results", help="Output directory for results")
    p.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    p.add_argument("--mc_samples", type=int, default=8, help="Monte Carlo samples per input")
    p.add_argument("--limit", type=int, default=0, help="Limit number of samples to process (0 for all)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    args = p.parse_args()
    main(args)