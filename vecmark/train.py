import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import math

from model import DimensionAwareModulator, UNet1D, DynamicDiffusion, VectorDataset

def build_loader(path, vector_len, batch_size, shuffle):
    """Build data loader from vector file"""
    ds = VectorDataset(path, vector_len=vector_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=min(4, os.cpu_count()))

def _to_tensor_stat(x, device):
    """Convert statistics to 1x1xL tensor format"""
    if isinstance(x, list):
        x = np.array(x)
    
    if isinstance(x, torch.Tensor):
        t = x.detach().float().to(device)
    else:
        t = torch.tensor(x, dtype=torch.float32, device=device)
    
    if t.ndim == 1:
        t = t.view(1, 1, -1)
    elif t.ndim == 2: 
        t = t.view(1, *t.shape) if t.shape[0] == 1 else t.unsqueeze(0)
    return t

def calculate_data_stats(dataloader):
    """Calculate mean and std statistics from dataloader"""
    all_vectors = []
    for batch in dataloader:
        all_vectors.append(batch)
    
    all_vectors = torch.cat(all_vectors, dim=0)
    mean_val = all_vectors.mean(dim=0)
    std_val = all_vectors.std(dim=0)
    
    if hasattr(mean_val, 'numpy'):
        mean_val = mean_val.numpy()
        std_val = std_val.numpy()
    
    return {
        'mean': mean_val.tolist() if hasattr(mean_val, 'tolist') else mean_val,
        'std': std_val.tolist() if hasattr(std_val, 'tolist') else std_val,
        'vector_len': all_vectors.shape[1]
    }

def save_checkpoint(model, diffusion, optimizer, epoch, loss, stats, save_dir, config=None):
    """Save training checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_state': model.state_dict(),
        'modulator_state': diffusion.noise_modulator.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'stats': stats,
        'config': config
    }
    save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, save_path)
    return save_path

def get_config_for_dimension(vector_len, manual_args=None, scale_factor=1.0):
    """Get dimension-based configuration with manual overrides and scaling"""
    config = {
        'base_channels': max(64, int(64 * (vector_len / 384))),
        'num_layers': min(7, max(4, int(math.log2(vector_len)) - 1)),
        'modulator_hidden': max(64, int(64 * (vector_len / 384))),
        'noise_steps': min(2000, max(800, int(1000 * (vector_len / 384)))),
        'beta_start': 1e-4 * (384 / vector_len),
        'beta_end': 0.02 * (vector_len / 384)
    }
    
    if vector_len >= 1024:
        config.update({
            'base_channels': 128,
            'num_layers': 7,
            'modulator_hidden': 128,
            'noise_steps': 1500
        })
    elif vector_len <= 256:
        config.update({
            'base_channels': 48,
            'num_layers': 4,
            'modulator_hidden': 48,
            'noise_steps': 800
        })
    
    if manual_args:
        for key in ['base_channels', 'num_layers', 'modulator_hidden']:
            if getattr(manual_args, key, None) is not None:
                config[key] = getattr(manual_args, key)
    
    config['base_channels'] = int(config['base_channels'] * scale_factor)
    config['modulator_hidden'] = int(config['modulator_hidden'] * scale_factor)
    
    return config

@torch.no_grad()
def anomaly_score_batch(model, diffusion, x, mc_samples=8, t_min=1, t_max=None, mean=None, std=None):
    """Calculate one-class anomaly score for batch with optional normalization"""
    model.eval()
    if mean is not None and std is not None:
        x = (x - mean) / torch.clamp(std, min=1e-6)

    B = x.size(0)
    T = diffusion.noise_steps if t_max is None else min(t_max, diffusion.noise_steps)
    scores = torch.zeros(B, device=x.device)
    for _ in range(mc_samples):
        t = torch.randint(low=max(1, t_min), high=T, size=(B,), device=x.device)
        x_t, mod_noise = diffusion.noise_images(x, t)
        pred_noise = model(x_t, t)
        mse = torch.mean((pred_noise - mod_noise) ** 2, dim=(1,2))
        scores += mse
    scores = scores / float(mc_samples)
    return scores

def evaluate_threshold(model, diffusion, val_loader_a, val_loader_other, device, target_fpr=0.05, mc_samples=8, mean=None, std=None):
    """Evaluate and select optimal threshold using A and optionally OTHER classes"""
    model.eval()

    scores_a = []
    for xb in val_loader_a:
        xb = xb.to(device).unsqueeze(1)
        scores_a.append(anomaly_score_batch(model, diffusion, xb, mc_samples=mc_samples, mean=mean, std=std).cpu())
    scores_a = torch.cat(scores_a).numpy()

    thr_default = float(np.quantile(scores_a, 1.0 - target_fpr))

    summary = {
        "target_fpr": target_fpr,
        "thr_default": thr_default,
        "n_a": int(scores_a.shape[0]),
        "a_mean": float(scores_a.mean()),
        "a_std": float(scores_a.std() + 1e-12),
    }

    if val_loader_other is None:
        return thr_default, summary

    scores_b = []
    for xb in val_loader_other:
        xb = xb.to(device).unsqueeze(1)
        scores_b.append(anomaly_score_batch(model, diffusion, xb, mc_samples=mc_samples, mean=mean, std=std).cpu())
    scores_b = torch.cat(scores_b).numpy()
    summary.update({
        "n_other": int(scores_b.shape[0]),
        "other_mean": float(scores_b.mean()),
        "other_std": float(scores_b.std() + 1e-12),
    })

    qs = np.linspace(0.50, 0.999, 50)
    best_thr, best_J, best_stats = thr_default, -1.0, None
    for q in qs:
        thr = float(np.quantile(scores_a, q))
        tpr = float((scores_b >= thr).mean())
        fpr = float((scores_a >= thr).mean())
        J = tpr - fpr
        if J > best_J:
            best_J = J
            best_thr = thr
            best_stats = {"quantile": float(q), "TPR_other": tpr, "FPR_a": fpr}

    summary["threshold_search"] = best_stats
    return best_thr, summary

def train_oneclass(args):
    """One-class training procedure with normalization and threshold selection"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[oneclass] Using device: {device}")
    
    dim_config = get_config_for_dimension(
        args.vector_len, 
        manual_args=args,
        scale_factor=args.scale_factor
    )
    args = argparse.Namespace(**{**vars(args), **dim_config})
    
    print("=" * 50)
    print(f"One-Class Training Configuration for {args.vector_len}D vectors:")
    print(f"- Base Channels: {args.base_channels}")
    print(f"- Number of Layers: {args.num_layers}")
    print(f"- Modulator Hidden: {args.modulator_hidden}")
    print(f"- Noise Steps: {args.noise_steps}")
    print(f"- Beta Schedule: {args.beta_start:.2e} to {args.beta_end:.4f}")
    print("=" * 50)

    train_loader_a = build_loader(args.train_a, args.vector_len, args.batch_size, shuffle=True)

    stats = calculate_data_stats(train_loader_a)
    mean = _to_tensor_stat(stats["mean"], device)
    std = torch.clamp(_to_tensor_stat(stats["std"], device), min=1e-6)
    print(f"Train A: {len(train_loader_a.dataset)} vectors, len={args.vector_len}")

    model = UNet1D(
        vector_len=args.vector_len,
        base_channels=args.base_channels,
        num_layers=args.num_layers
    ).to(device)
    
    diffusion = DynamicDiffusion(
        noise_steps=args.noise_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        vector_len=args.vector_len,
        device=device,
        modulator_hidden=args.modulator_hidden
    )

    optimizer = optim.AdamW(
        list(model.parameters()) + list(diffusion.noise_modulator.parameters()), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=args.patience
    )

    best_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    config_path = os.path.join(args.save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'vector_len': args.vector_len,
            'base_channels': args.base_channels,
            'num_layers': args.num_layers,
            'modulator_hidden': args.modulator_hidden,
            'noise_steps': args.noise_steps,
            'beta_start': args.beta_start,
            'beta_end': args.beta_end,
            'device': args.device
        }, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        diffusion.noise_modulator.train()
        
        total_loss = 0.0
        recon_loss_sum = 0.0
        
        pbar = tqdm(train_loader_a, desc=f"Epoch {epoch}/{args.epochs}")
        for xb in pbar:
            xb = xb.to(device).unsqueeze(1)
            xb = (xb - mean) / std

            t = diffusion.sample_timesteps(xb.shape[0]).to(device)
            loss, recon_loss, imp_loss = diffusion.joint_train_step(model, xb, t, optimizer)
            
            total_loss += loss
            recon_loss_sum += recon_loss
            
            pbar.set_postfix(
                loss=f"{loss:.4f}", 
                recon=f"{recon_loss:.4f}"
            )

        avg_total = total_loss / len(train_loader_a)
        avg_recon = recon_loss_sum / len(train_loader_a)
        
        scheduler.step(avg_total)
        print(f"[Epoch {epoch}] Total={avg_total:.6f} | Recon={avg_recon:.6f} | "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

        if epoch % args.save_interval == 0:
            torch.save({
                'model_state': model.state_dict(),
                'modulator_state': diffusion.noise_modulator.state_dict(),
                'stats': stats,
                'config': {
                    'vector_len': args.vector_len,
                    'base_channels': args.base_channels,
                    'num_layers': args.num_layers,
                    'modulator_hidden': args.modulator_hidden,
                    'noise_steps': args.noise_steps,
                    'beta_start': args.beta_start,
                    'beta_end': args.beta_end
                }
            }, os.path.join(args.save_dir, f"checkpoint_{epoch}.pt"))

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'model_state': model.state_dict(),
                'modulator_state': diffusion.noise_modulator.state_dict(),
                'stats': stats,
                'config': {
                    'vector_len': args.vector_len,
                    'base_channels': args.base_channels,
                    'num_layers': args.num_layers,
                    'modulator_hidden': args.modulator_hidden,
                    'noise_steps': args.noise_steps,
                    'beta_start': args.beta_start,
                    'beta_end': args.beta_end
                }
            }, os.path.join(args.save_dir, "best_model.pt"))
            print("Saved best model.")

    val_loader_a = build_loader(args.val_a, args.vector_len, args.batch_size, shuffle=False)
    val_loader_other = build_loader(args.val_other, args.vector_len, args.batch_size, shuffle=False) if args.val_other else None

    best_ckpt = torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt['model_state'])
    diffusion.noise_modulator.load_state_dict(best_ckpt['modulator_state'])

    thr, thr_summary = evaluate_threshold(
        model, diffusion, val_loader_a, val_loader_other, device,
        target_fpr=args.target_fpr, mc_samples=args.mc_samples,
        mean=mean, std=std
    )
    
    print("Selected threshold:", thr)
    print("Summary:", thr_summary)

    package = {
        "model_state": model.state_dict(),
        "modulator_state": diffusion.noise_modulator.state_dict(),
        "vector_len": args.vector_len,
        "noise_steps": args.noise_steps,
        "threshold": thr,
        "threshold_summary": thr_summary,
        "stats": {
            **stats,
            "mean": stats["mean"].tolist() if hasattr(stats["mean"], 'tolist') else stats["mean"],
            "std": stats["std"].tolist() if hasattr(stats["std"], 'tolist') else stats["std"],
        },
        "config": {
            'vector_len': args.vector_len,
            'base_channels': args.base_channels,
            'num_layers': args.num_layers,
            'modulator_hidden': args.modulator_hidden,
            'noise_steps': args.noise_steps,
            'beta_start': args.beta_start,
            'beta_end': args.beta_end
        }
    }
    torch.save(package, os.path.join(args.save_dir, "oneclass_model.pt"))
    
    with open(os.path.join(args.save_dir, "oneclass_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "threshold": float(thr),
            "summary": thr_summary,
            "vector_len": args.vector_len,
            "normalization": {
                "mean": stats["mean"].tolist() if isinstance(stats["mean"], np.ndarray) else stats["mean"],
                "std": stats["std"].tolist() if isinstance(stats["std"], np.ndarray) else stats["std"]
            },
            "config": {
                'vector_len': args.vector_len,
                'base_channels': args.base_channels,
                'num_layers': args.num_layers,
                'modulator_hidden': args.modulator_hidden,
                'noise_steps': args.noise_steps,
                'beta_start': args.beta_start,
                'beta_end': args.beta_end
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Saved one-class model to {os.path.join(args.save_dir, 'oneclass_model.pt')}")

def parse_args():
    """Parse command line arguments for one-class diffusion model training"""
    p = argparse.ArgumentParser(description="Train one-class diffusion model with dynamic noise, normalization and anomaly scoring")
    
    p.add_argument('--train_a', type=str, required=True, help="A class training data (.pkl/.npy)")
    p.add_argument('--val_a', type=str, required=True, help="A class validation data (.pkl/.npy)")
    p.add_argument('--val_other', type=str, default=None, help="Other encoder validation data (optional)")
    p.add_argument('--vector_len', type=int, default=2048, help="Input vector dimension")
    p.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    p.add_argument('--batch_size', type=int, default=64, help="Batch size")
    p.add_argument('--noise_steps', type=int, default=1500, help="Number of diffusion steps")
    p.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    p.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    p.add_argument('--patience', type=int, default=5, help="Learning rate scheduler patience")
    p.add_argument('--target_fpr', type=float, default=0.05, help="Target false positive rate")
    p.add_argument('--mc_samples', type=int, default=4, help="Number of Monte Carlo samples")
    p.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help="Training device")
    p.add_argument('--save_dir', type=str, default='./model/out_oneclass', help="Model save directory")
    p.add_argument('--save_interval', type=int, default=10, help="Checkpoint save interval")
    p.add_argument('--base_channels', type=int, default=None, help="Manual base channels setting")
    p.add_argument('--num_layers', type=int, default=None, help="Manual number of layers setting")
    p.add_argument('--modulator_hidden', type=int, default=None, help="Manual modulator hidden size setting")
    p.add_argument('--scale_factor', type=float, default=1.0, help="Model scaling factor (0.5-1.0)")

    return p.parse_args()

def main():
    """Main training entry point"""
    args = parse_args()
    
    print("=" * 50)
    print(f"Starting oneclass training")
    print(f"Vector length: {args.vector_len}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 50)
    
    try:
        train_oneclass(args)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("Training completed successfully!")

if __name__ == "__main__":
    main()