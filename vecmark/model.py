import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle

class VectorDataset(torch.utils.data.Dataset):
    """Text embedding vector dataset"""
    
    def __init__(self, data_path, vector_len=384):
        self.vectors = self.load_vectors(data_path)
        self.vector_len = vector_len
        
    def load_vectors(self, path):
        """Load embedding vectors from file"""
        if path.endswith('.pkl'):
            vectors = []
            with open(path, 'rb') as f:
                vectors = pickle.load(f)
            return np.array(vectors)
        elif path.endswith('.npy'):
            return np.load(path)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .npy")
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        return torch.tensor(self.vectors[idx], dtype=torch.float32)
    
class DimensionAwareModulator(nn.Module):
    """Dynamic noise modulator with dimension importance evaluation"""
    
    def __init__(self, vector_len=384, hidden_dim=64):
        super().__init__()
        self.vector_len = vector_len
        
        hidden_dim = max(32, min(128, hidden_dim))
        
        self.dim_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            ) for _ in range(vector_len)
        ])
        
        self.importance_evaluator = nn.Sequential(
            nn.Linear(vector_len, 256),
            nn.ReLU(),
            nn.Linear(256, vector_len),
            nn.Softmax(dim=1)
        )
    
    def forward(self, base_noise, x):
        """Apply dimension-aware modulation to base noise"""
        modulated_noise = torch.zeros_like(base_noise)
        for i, modulator in enumerate(self.dim_modulators):
            dim_feature = x[:, :, i:i+1]
            mod_coeff = modulator(dim_feature)
            modulated_noise[:, :, i] = mod_coeff.squeeze(-1) * base_noise[:, :, i]
        
        with torch.no_grad():
            target_std = base_noise.std(dim=2, keepdim=True) + 1e-6
            mod_std = modulated_noise.std(dim=2, keepdim=True) + 1e-6
        scale = target_std / mod_std
        return modulated_noise * scale

class UNet1D(nn.Module):
    """1D U-Net model for diffusion process"""
    
    def __init__(self, vector_len=384, time_dim=256, base_channels=64, num_layers=5):
        super().__init__()
        self.vector_len = vector_len
        self.time_dim = time_dim
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        max_downsample = int(math.log2(vector_len))
        self.downsample_layers = min(num_layers, max_downsample - 1)
        self.reduced_len = vector_len // (2 ** self.downsample_layers)
        
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, base_channels * (2 ** self.downsample_layers))
        )
        
        self.encoder = nn.ModuleList()
        current_channels = 1
        for i in range(self.downsample_layers):
            out_channels = base_channels * (2 ** i)
            kernel_size = min(7, vector_len // max(1, (2 ** i)))
            
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, out_channels, 
                             kernel_size=kernel_size, padding=kernel_size//2),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            )
            current_channels = out_channels
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(current_channels, current_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(current_channels * 2, current_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        current_channels = current_channels * 2
        
        self.decoder = nn.ModuleList()
        for i in range(self.downsample_layers):
            idx = self.downsample_layers - i - 1
            skip_channels = base_channels * (2 ** idx)
            in_channels = current_channels + skip_channels
            out_channels = skip_channels
            
            kernel_size = min(7, vector_len // max(1, (2 ** idx)))
            kernel_size = max(1, kernel_size)
            
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels, 
                        out_channels, 
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size//2,
                        output_padding=1
                    ),
                    nn.ReLU()
                )
            )
            current_channels = out_channels
        
        self.final_conv = nn.Conv1d(current_channels, 1, kernel_size=1)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, x, t):
        orig_len = x.shape[-1]
        
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        t_emb = self.time_embed(t_emb)
        
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        
        x = self.bottleneck(x)
        
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, x.shape[2])
        x = x + t_emb
        
        for i, layer in enumerate(self.decoder):
            skip_idx = self.downsample_layers - i - 1
            skip = skips[skip_idx]
            
            if skip.shape[2] > x.shape[2]:
                diff = skip.shape[2] - x.shape[2]
                x = F.pad(x, (0, diff), "constant", 0)
            elif skip.shape[2] < x.shape[2]:
                diff = x.shape[2] - skip.shape[2]
                skip = F.pad(skip, (0, diff), "constant", 0)
                
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
        
        x = self.final_conv(x)
        
        if x.shape[-1] != orig_len:
            diff = orig_len - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff), "constant", 0)
            else:
                x = x[:, :, :orig_len]
                
        return x

class DynamicDiffusion:
    """Dynamic noise diffusion process controller with parameterized configuration"""
    
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, 
                 vector_len=384, device="cuda", modulator_hidden=64):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.vector_len = vector_len
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.noise_modulator = DimensionAwareModulator(
            vector_len=vector_len, 
            hidden_dim=modulator_hidden
        ).to(device)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        """Add noise to vectors with dynamic modulation"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        
        base_noise = torch.randn_like(x)
        mod_noise = self.noise_modulator(base_noise, x)
        noisy_x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * mod_noise
        
        return noisy_x, mod_noise
    
    def joint_train_step(self, model, x, t, optimizer):
        """Joint training step for noise modulator and denoising model"""
        x_t, mod_noise = self.noise_images(x, t)
        pred_noise = model(x_t, t)
        reconstruction_loss = F.mse_loss(pred_noise, mod_noise)
        
        with torch.no_grad():
            importance = self.noise_modulator.importance_evaluator(x.squeeze(1))
        
        pred_errors = (pred_noise - mod_noise).abs().mean(dim=1)
        importance_loss = F.mse_loss(pred_errors, importance)
        total_loss = reconstruction_loss + 1e-5 * importance_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), reconstruction_loss.item(), importance_loss.item()
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

def get_config_for_dimension(vector_len):
    """Get recommended configuration based on vector dimension"""
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
        
    return config