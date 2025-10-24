import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import DAE, VAE, AAE
from vocab import Vocab
from utils import set_seed, logging, load_sent
import os
import matplotlib.pyplot as plt
from scipy.stats import kstest
import sys

if np.__version__ < '2.0.0':
    # For NumPy 1.x, map numpy.core to numpy._core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._exceptions'] = getattr(np.core, '_exceptions', None)


def load_and_sample_test_data(pkl_path, sample_size):
    """Load data from test.pkl and randomly sample the specified number of samples."""
    embeddings = []
    with open(pkl_path, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                embeddings.append(data[1])
            except EOFError:
                break
    
    # convert to Tensor
    embeddings = torch.stack([torch.as_tensor(e) for e in embeddings]) 
    indices = torch.randperm(len(embeddings))[:sample_size]
    sampled_embeddings = embeddings[indices]  
    return sampled_embeddings

def preprocess_batches(embeddings, batch_size):
    batches = torch.split(embeddings, batch_size)
    batches = torch.stack(batches)

    return batches

def compute_test_vector_loss(model, batches, device, shuffle_dims=False):
    model.eval()
    all_test_vector_losses = []

    with torch.no_grad():
        for i, batch in enumerate(batches):
            inputs = batch.unsqueeze(0).to(device)
            
            # model inference
            inputs_for_model = inputs.clone()  
            mu, logvar, z, reconstructed = model(inputs_for_model, is_train=False)
            vector_diffs = inputs_for_model - reconstructed
            vector_diffs = vector_diffs.squeeze(0)  
            all_test_vector_losses.append(vector_diffs.cpu().numpy())

    test_vector_losses = np.concatenate(all_test_vector_losses, axis=0)  

    # compute mean squared error for each sample
    loss_magnitudes = np.mean(np.square(test_vector_losses), axis=1)  # MSE of all loss vectors
    print(f"Final test_vector_losses_mse shape: {loss_magnitudes.shape}") 

    return loss_magnitudes

def get_model(model_path):
    """load pretrained model"""
    # load checkpoint
    ckpt = torch.load(model_path, map_location='cpu')
    train_args = ckpt['args']

    # load vocab
    vocab_file = os.path.join(os.path.dirname(model_path), 'vocab.txt')
    vocab = Vocab(vocab_file)

    # create model
    model = VAE(vocab, train_args)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    
    return model

device = torch.device('cpu')

# Load model
model = get_model("768_train/model.pt")


# ------

# Load training data (wiki-allmini) and compute losses
with open("data/768/wikitext_bge-base-en-v1.5.pkl", "rb") as f:
    loaded_data = pickle.load(f)[:1000]

train_embeddings = torch.stack([torch.as_tensor(e) for e in loaded_data])  
print(f"Training embeddings shape: {train_embeddings.shape}")
batches = preprocess_batches(train_embeddings, batch_size=10)
train_vector_losses = compute_test_vector_loss(model, batches, device, shuffle_dims=False)

# nfcorpus-allmini: calculate test losses
with open("data/768/quora_bge-base-en-v1.5.pkl", "rb") as f:
    loaded_data = pickle.load(f)[:1000]

test_embeddings = torch.stack([torch.as_tensor(e) for e in loaded_data]) 
print(f"Test embeddings shape: {test_embeddings.shape}")
batches = preprocess_batches(test_embeddings, batch_size=10)
test_vector_losses = compute_test_vector_loss(model, batches, device, shuffle_dims=False)


test = test_vector_losses
train = train_vector_losses

# initialization
A = test
B = train

# KS test
statistic,p_value =kstest(A, B)


print(f"KS Statistic: {statistic:.6f}")
print(f"P-value: {np.format_float_scientific(p_value, precision=15)}")


plt.figure(figsize=(16, 5))

# visualization
# plt.subplot(1, 2, 1)
plt.hist(A, bins=50, alpha=0.7, label='Dataset quora_bge')
plt.hist(B, bins=50, alpha=0.7, label='Dataset train')
plt.xlabel('Vector Loss Value')
plt.ylabel('Density')
plt.title(f'KS Test Comparison\n Statistic = {statistic:.10e}, p-value = {p_value:.10e}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

