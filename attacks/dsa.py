import torch
import numpy as np
import pickle
import sys

if np.__version__ < '2.0.0':
    # For NumPy 1.x, map numpy.core to numpy._core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._exceptions'] = getattr(np.core, '_exceptions', None)

# Training phase

# Data loading function
# Data loading
def load_embedding_pkl(path):
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
        embeddings = embeddings[:1000]
    return embeddings

# Load test data
print("Loading test embeddings...")
test_path = 'data/2048/nfcorpus_gemma-2b-embeddings.pkl'
test_embeddings_list = load_embedding_pkl(test_path)
test_embeddings = torch.stack([torch.tensor(x) for x in test_embeddings_list])

B = 1000
indices = np.random.choice(len(test_embeddings), B, replace=False)
test_batch = test_embeddings[indices]
print("Selected test data, shape:", test_batch.shape) 


seed = 42  
torch.manual_seed(seed)
np.random.seed(seed)


# Randomly permute all dimensions
perm = torch.randperm(2048) 
print("Random permutation generated")
shuffled_batch = test_batch[:, perm]  # Randomly reorder dimensions for each sample
print("Dimension-wise random shuffle completed, shape:", shuffled_batch.shape)  


# Save directly
# Save as pkl format
with open("2048_dsa/dsa_2048-nfcorpus_gemma-2b-embeddings.pkl", "wb") as f:
    pickle.dump(shuffled_batch.numpy(), f)
print("Shuffled test data saved as pkl")

# load to verify
with open("2048_dsa/dsa_2048-nfcorpus_gemma-2b-embeddings.pkl", "rb") as f:
    loaded_data = pickle.load(f)
print(f"Verified loaded data shape: {loaded_data.shape}")

