import pickle
import numpy as np
import sys

if np.__version__ < '2.0.0':
    # For NumPy 1.x, map numpy.core to numpy._core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._exceptions'] = getattr(np.core, '_exceptions', None)

# Data loading function
def load_embedding_pkl(path):
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
        embeddings = embeddings[:1000]
    return embeddings


# Add Gaussian noise to embeddings
def add_gaussian_noise(embeddings, noise_ratio=0.1):
    embeddings = np.array(embeddings, dtype=np.float32)  
    noise = np.random.randn(*embeddings.shape) * (embeddings * noise_ratio)
    return embeddings + noise


if __name__ == "__main__":
    path = "data/2048/scidocs_seed.pkl"
    embeddings = load_embedding_pkl(path)    
    noisy_embeddings = add_gaussian_noise(embeddings, noise_ratio=0.1)
    print("Original shape:", embeddings.shape)
    print("Noisy shape:", noisy_embeddings.shape)
    # Save the noisy results
    # Save as pkl format
    with open("data/2048_gs/gs_scidocs_seed.pkl", "wb") as f:
        pickle.dump(noisy_embeddings, f)
    print("Gaussian-noised test data saved as pkl")

    # Load verification data
    with open("data/2048_gs/gs_scidocs_seed.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    print(f"Verified loaded data shape: {loaded_data.shape}")
