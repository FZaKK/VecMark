import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
import pickle
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

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
    return embeddings

# Load data
print("Loading training embeddings...")
train_path = 'data/2048/nfcorpus_seed.pkl'
train_embeddings_list = load_embedding_pkl(train_path)
train_embeddings = torch.stack([torch.tensor(x) for x in train_embeddings_list])
print("Training embeddings shape:", train_embeddings.shape)


train_embeddings_np = train_embeddings.numpy()  
dim_distributions = train_embeddings_np.T  
print("Per-dimension distribution shape:", dim_distributions.shape)  

# For each dimension, build a PDF and use histogram as clustering features
hist_features = []
gaussian_pdfs = []

for i in range(2048):
    values = dim_distributions[i]
    mu = np.mean(values)
    std = np.std(values)
    if std < 1e-6:  # avoid zero standard deviation
        std = 1e-6
    pdf = norm(loc=mu, scale=std)
    gaussian_pdfs.append(pdf)

    hist, _ = np.histogram(values, bins=50, range=(-0.3, 0.3), density=True)
    hist_features.append(hist)

hist_features = np.stack(hist_features)  
print("Clustering feature shape:", hist_features.shape)  
print("Number of pdfs:", len(gaussian_pdfs))  

# Clustering (20 clusters)
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(hist_features)  # which cluster each dimension belongs to
print("Cluster labels shape:", cluster_labels.shape)  

# Build aggregate PDF for each cluster (combine values of dims in that cluster)
cluster_pdfs = []

for c in range(n_clusters):
    dim_indices = np.where(cluster_labels == c)[0]
    if len(dim_indices) == 0:
        # avoid empty-cluster errors
        cluster_pdfs.append(None)
        continue
    combined_values = dim_distributions[dim_indices].flatten()
    mu = np.mean(combined_values)
    std = np.std(combined_values)
    if std < 1e-6:  
        std = 1e-6
    pdf = norm(loc=mu, scale=std)
    cluster_pdfs.append(pdf)

print("Number of cluster PDFs:", len(cluster_pdfs))  # 20

# Save: cluster_pdfs, kdes, cluster_labels

# Testing phase

# Load test data
print("Loading test embeddings...")
test_path = 'data/2048/scidocs_seed.pkl'
test_embeddings_list = load_embedding_pkl(test_path)
test_embeddings = torch.stack([torch.tensor(x) for x in test_embeddings_list])
B = 1000
indices = np.random.choice(len(test_embeddings), B, replace=False)
test_batch = test_embeddings[indices]
print("Selected test data, shape:", test_batch.shape) 

# Dimension recovery
B, D = test_batch.shape  # B is batch size
print("Test data shape:", B, D)
test_embeddings_np = test_batch.numpy()
test_distributions = test_embeddings_np.T 
print("Test per-dimension distribution shape:", test_distributions.shape) 

n_clusters = len(cluster_pdfs)

# Record: for each cluster -> contained original dimension indices
cluster_to_dims = defaultdict(list)
for i, c in enumerate(cluster_labels):
    cluster_to_dims[c].append(i)
print("Length:", len(cluster_to_dims))  
"""
cluster_to_dims = {
    0: [0, 3],  # Cluster 0 contains dimensions 0 and 3
    1: [1, 2],  # Cluster 1 contains dimensions 1 and 2
    2: [4]      # Cluster 2 contains dimension 4
}
"""

# For each test dimension -> find the most likely cluster
test_dim_to_cluster = []

for i in range(D):
    # print(f"Processing dimension {i+1}...")
    values = test_distributions[i]  

    # First determine which cluster it most likely belongs to
    probs = [pdf.pdf(values).mean() for pdf in cluster_pdfs]  
    normalized_probs = probs / np.sum(probs)
    cluster_id = np.argmax(normalized_probs)
    test_dim_to_cluster.append(cluster_id)
print("Length of test-dimension-to-cluster mapping:", len(test_dim_to_cluster))  

# Within each cluster, perform Hungarian matching
recovered_indices = [None] * D
used_test_dims = set()
used_ref_dims = set()

unmatched_test_dims = []

for c in range(n_clusters):
    print(f"Processing cluster {c}...")
    # Find test dims (shuffled) and reference dims (belonging to this cluster)
    test_indices = [i for i, cid in enumerate(test_dim_to_cluster) if cid == c]
    ref_indices = cluster_to_dims[c]

    M = len(test_indices)
    N = len(ref_indices)
    print(f"Cluster {c}: #test dims = {M}, #reference dims = {N}")
            
    if M == 0 or N == 0:
        unmatched_test_dims.extend(test_indices)
        continue

    # Build cost matrix
    cost_matrix = np.zeros((M, N))
    for i in range(M):
        values = test_distributions[test_indices[i]]
        for j in range(N):
            cost_matrix[i, j] = -gaussian_pdfs[ref_indices[j]].pdf(values).mean()

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = set()
    for i, j in zip(row_ind, col_ind):
        test_idx = test_indices[i]
        ref_idx = ref_indices[j]
        if recovered_indices[ref_idx] is None and test_idx not in used_test_dims:
            recovered_indices[ref_idx] = test_idx
            used_ref_dims.add(ref_idx)
            used_test_dims.add(test_idx)
            matched.add(test_idx)

    print(f"Cluster {c} matching done, matches: {len(matched)}")

    # Add unmatched test dimensions to the fill-in list
    for test_idx in test_indices:
        if test_idx not in matched:
            unmatched_test_dims.append(test_idx)

    print("Number of unmatched test dimensions:", len(unmatched_test_dims))  # remaining unmatched dims

# Global greedy compensation
all_dims = set(range(D))
unused_ref_dims = list(all_dims - used_ref_dims)
print("Number of unused reference dimensions:", len(unused_ref_dims))  # remaining unused reference dims

for i, test_idx in enumerate(unmatched_test_dims):
    # print(f"Global compensation handling unmatched test dim #{i+1}...")
    values = test_distributions[test_idx]
    best_score = -np.inf
    best_ref = None
    for ref_idx in unused_ref_dims:
        score = gaussian_pdfs[ref_idx].pdf(values).mean()
        if score > best_score:
            best_score = score
            best_ref = ref_idx
    recovered_indices[best_ref] = test_idx  
    unused_ref_dims.remove(best_ref)


# Length of recovered dimension indices
print("Length of recovered dimension indices:", len(recovered_indices))
# Restore dimension order
restored_embeddings = test_batch[:, recovered_indices]
print("Restored test data shape:", restored_embeddings.shape)  

restored_embeddings_np = restored_embeddings.numpy()

# Save directly
# Save as pkl format
with open("2048_sword/sword_scidocs_seed.pkl", "wb") as f:
    pickle.dump(restored_embeddings_np, f)
print("Restored test data saved as pkl")

# Load and verify
with open("2048_sword/sword_scidocs_seed.pkl", "rb") as f:
    loaded_data = pickle.load(f)
print(f"Verified loaded data shape: {loaded_data.shape}")
