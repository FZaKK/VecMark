import numpy as np
import pickle

import sys

if np.__version__ < '2.0.0':
    # For NumPy 1.x, map numpy.core to numpy._core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._exceptions'] = getattr(np.core, '_exceptions', None)

# Data loading function
def load_embedding_pkl(path, n=100, seed=42):
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)           
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embeddings), n, replace=False)
    embeddings = np.array(embeddings)[idx]
    return embeddings

# Load and randomly sample 100 entries
sea_path = 'data/2048/scidocs_seed.pkl'
sea_embeddings = load_embedding_pkl(sea_path, n=100, seed=42)
print("Loaded sea shape:", np.array(sea_embeddings).shape)


# Save directly
# Save as pkl format
with open("2048_sea/sea_scidocs_seed.pkl", "wb") as f:
    pickle.dump(sea_embeddings, f)
print("Extracted test data saved as pkl")

# Load verification data
with open("2048_sea/sea_scidocs_seed.pkl", "rb") as f:
    loaded_data = pickle.load(f)
print(f"Verified loaded data shape: {loaded_data.shape}")
