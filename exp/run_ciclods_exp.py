import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import scipy.sparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scipy
import scipy.sparse
import scanpy as sc
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# 1) POSITIONAL ENCODING (your version)
# ==========================================
def positional_encoding_2d_wmode(coords, total_dims, mode="interleaved"):
    assert mode in ["separate", "interleaved", "norm", "interaction"], \
        "Mode must be 'separate', 'interleaved', 'norm', or 'interaction'."
    assert total_dims % 2 == 0, "total_dims must be even."

    d = total_dims // 2
    if mode == "norm":
        frequencies = np.power(10000, -np.arange(0, total_dims, 1) / total_dims)
    else:
        frequencies = np.power(10000, -np.arange(0, total_dims, 2) / total_dims)

    x_coords, y_coords = coords[:, 0], coords[:, 1]

    x_sine = np.sin(np.outer(x_coords, frequencies))
    x_cosine = np.cos(np.outer(x_coords, frequencies))
    y_sine = np.sin(np.outer(y_coords, frequencies))
    y_cosine = np.cos(np.outer(y_coords, frequencies))

    if mode == "separate":
        x_encoded = np.hstack([x_sine[:, :d], x_cosine[:, :d]])
        y_encoded = np.hstack([y_sine[:, :d], y_cosine[:, :d]])
        encoded_coords = np.hstack([x_encoded, y_encoded])

    elif mode == "interleaved":
        interleaved = np.empty((coords.shape[0], total_dims), dtype=np.float64)
        interleaved[:, 0::4] = x_sine[:, :d // 2]
        interleaved[:, 1::4] = y_sine[:, :d // 2]
        interleaved[:, 2::4] = x_cosine[:, :d // 2]
        interleaved[:, 3::4] = y_cosine[:, :d // 2]
        encoded_coords = interleaved

    elif mode == "norm":
        magnitude = np.sqrt(x_sine**2 + x_cosine**2 + y_sine**2 + y_cosine**2)
        encoded_coords = magnitude[:, :total_dims]

    elif mode == "interaction":
        encoded_coords_cos = (x_cosine * y_cosine)
        encoded_coords_sin = (x_sine * y_sine)
        encoded_coords = np.zeros((encoded_coords_cos.shape[0], encoded_coords_cos.shape[1] * 2))
        encoded_coords[:, 0::2] = encoded_coords_cos
        encoded_coords[:, 1::2] = encoded_coords_sin

    return encoded_coords



# ==========================================
# 1. HELPER: POSITIONAL ENCODING
# ==========================================

def positional_encoding_2d_wmode(coords, total_dims, mode="separate"):
    """
    Encodes 2D (x, y) coordinates into a higher-dimensional space using positional encoding.

    Parameters:
    - coords: 2D array (n_samples, 2), input 2D coordinates.
    - total_dims: int, the desired dimensionality of the output encoding.
    - mode: str, the encoding method. Options are "separate", "interleaved", or "norm".

    Returns:
    - encoded_coords: 2D array (n_samples, total_dims), positional-encoded features.
    """
    assert mode in ["separate", "interleaved", "norm",'interaction'], "Mode must be 'separate', 'interleaved', or 'norm'."
    assert total_dims % 2 == 0, "total_dims must be even."

    d = total_dims // 2  # Half of the dimensions for x and y (only applies to some modes)
    if mode == "norm":
        frequencies = np.power(10000, -np.arange(0, total_dims, 1) / total_dims)
    else:
        frequencies = np.power(10000, -np.arange(0, total_dims, 2) / total_dims)


    # Separate x and y coordinates
    x_coords, y_coords = coords[:, 0], coords[:, 1]

    # Compute sinusoidal encodings for x and y
    x_sine = np.sin(np.outer(x_coords, frequencies))
    x_cosine = np.cos(np.outer(x_coords, frequencies))
    y_sine = np.sin(np.outer(y_coords, frequencies))
    y_cosine = np.cos(np.outer(y_coords, frequencies))

    if mode == "separate":
        # Concatenate x and y encodings side by side
        x_encoded = np.hstack([x_sine[:, :d], x_cosine[:, :d]])
        y_encoded = np.hstack([y_sine[:, :d], y_cosine[:, :d]])
        encoded_coords = np.hstack([x_encoded, y_encoded])

    elif mode == "interleaved":
        # Interleave x and y encodings
        interleaved = np.empty((coords.shape[0], total_dims), dtype=np.float64)
        interleaved[:, 0::4] = x_sine[:, :d//2]
        interleaved[:, 1::4] = y_sine[:, :d//2]
        interleaved[:, 2::4] = x_cosine[:, :d//2]
        interleaved[:, 3::4] = y_cosine[:, :d//2]
        encoded_coords = interleaved

    elif mode == "norm":
        # Use total_dims directly for magnitude
        magnitude = np.sqrt(x_sine**2 + x_cosine**2 + y_sine**2 + y_cosine**2)
        encoded_coords = magnitude[:, :total_dims]  # Ensure the shape matches total_dims
    elif mode == "interaction":
        # Multiplicative interaction (creates grid-like interference patterns)
        # Useful for capturing joint X-Y dependencies
        # encoded_coords = (x_sine * y_sine)
        encoded_coords_cos = (x_cosine * y_cosine)
        encoded_coords_sin = (x_sine * y_sine)
        encoded_coords = np.hstack([encoded_coords_cos, encoded_coords_sin])
        encoded_coords = np.hstack([encoded_coords, encoded_coords_cos])
        encoded_coords = np.zeros((encoded_coords_cos.shape[0], encoded_coords_cos.shape[1]*2))
        encoded_coords[:, 0::2] = encoded_coords_cos
        encoded_coords[:, 1::2] = encoded_coords_sin
    return encoded_coords

# ==========================================
# 2. MODEL 1: CiCLODS (Yours)
# # ==========================================
def run_ciclods_model(adata_in, Ky, Kx, max_iter=100, use_pe=True, pe_weight=0.5):
    print(f"--- Running CiCLODS (Kx={Kx}) ---")
    adata = adata_in.copy()

    # Preprocessing (Model Specific)
    # sc.pp.filter_cells(adata, min_counts=30)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # Note: No log1p here, as per your preference for this model

    if scipy.sparse.issparse(adata.X):
        M_raw = adata.X.toarray()
    else:
        M_raw = adata.X.copy()

    # Model Normalization (M / std)
    col_var = np.var(M_raw, axis=0, ddof=1)
    valid_indices = np.where(col_var > 0)[0]
    M = M_raw[:, valid_indices]
    col_var = col_var[valid_indices]
    M = M / np.sqrt(col_var)
    gene_names = np.array(adata.var_names)[valid_indices]

    # PE Injection
    if use_pe:
        if M.shape[1] % 2 != 0:
            M = M[:, :-1]
            gene_names = gene_names[:-1]

        spatial_coords = adata.obsm['spatial']
        spatial_scaled = MinMaxScaler().fit_transform(spatial_coords) * np.pi
        print(M.shape)
        encoded_feats = positional_encoding_2d_wmode(spatial_scaled, total_dims=M.shape[1], mode='interleaved')

        M = StandardScaler().fit_transform(M) + (StandardScaler().fit_transform(encoded_feats) * pe_weight)


    # Iterative Optimization
    np.random.seed(42)
    idx = np.random.randint(0, Ky, size=M.shape[0])
    I = np.random.permutation(M.shape[1])[:Kx]

    for i in range(max_iter):
        I_prev = I.copy()

        # Feature Selection
        var_sum = np.zeros(M.shape[1])
        for j in range(Ky):
            mask = (idx == j)
            if np.sum(mask) > 1:
                var_sum += np.var(M[mask], axis=0, ddof=1) * (np.sum(mask) - 1)
        I = np.argpartition(var_sum, Kx)[:Kx]

        # Clustering
        kmeans = KMeans(n_clusters=Ky, n_init=5, random_state=42)
        kmeans.fit(M[:, I])
        if np.array_equal(idx, kmeans.labels_): break
        idx = kmeans.labels_

    adata.obs['ciclods'] = idx.astype(str)
    selected_genes = gene_names[I]
    return adata, selected_genes


# ==========================================
# 3. MODEL 2: Variance-Matched PCA (Baseline 1)
# ==========================================
def run_variance_matched_baseline(adata_in, selected_genes, n_clusters):
    print(f"--- Running Variance-Matched PCA ---")
    adata = adata_in.copy()

    # Log-Norm for PCA
    # sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

    # Calculate Target Variance Ratio
    total_variance = adata.n_vars
    selected_mask = adata.var_names.isin(selected_genes)
    target_ratio = np.sum(selected_mask) / total_variance
    print(f"  > Target Variance: {target_ratio:.2%}")

    # Run PCA
    sc.tl.pca(adata, n_comps=150)

    # Find cutoff
    explained = adata.uns['pca']['variance_ratio']
    cumulative = np.cumsum(explained)
    n_pcs = np.searchsorted(cumulative, target_ratio) + 1
    if n_pcs > 150: n_pcs = 150

    print(f"  > Using top {n_pcs} PCs")

    # Cluster
    X_pca = adata.obsm['X_pca'][:, :n_pcs]
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    kmeans.fit(X_pca)

    adata.obs['matched_baseline'] = kmeans.labels_.astype(str)
    return adata, n_pcs


# ==========================================
# 4. MODEL 3: Raw K-Means (Baseline 2)
# ==========================================
def run_kmeans_baseline(adata_in, n_clusters):
    print("--- Running K-Means (No PCA) ---")
    adata = adata_in.copy()

    # Standard Log-Norm
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)

    # Scale (Z-score)
    # This is where NaNs usually happen if a gene has 0 variance
    # sc.pp.scale(adata, max_value=10)

    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X

    # CRITICAL FIX: Replace NaNs with 0
    # NaNs happen if scaling divided by 0 std dev (constant genes)
    if np.isnan(X).any():
        print("  > Warning: NaNs detected after scaling. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    kmeans.fit(X)

    adata.obs['kmeans_raw'] = kmeans.labels_.astype(str)
    return adata


# ==========================================
# 5. EZ PLOTTER (Metadata-Free)
# ==========================================
def plot_ez(adata, keys, titles):
    x = adata.obsm['spatial'][:, 1]
    y = -adata.obsm['spatial'][:, 0]  # Flip Y for correct orientation

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    if len(keys) == 1: axes = [axes]

    for i, key in enumerate(keys):
        ax = axes[i]
        labels = adata.obs[key].values
        unique = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))

        for label, color in zip(unique, colors):
            mask = (labels == label)
            ax.scatter(x[mask], y[mask], label=label, s=15, c=[color], alpha=1.0, edgecolors='none')

        ax.set_title(titles[i])
        ax.axis('off')
        if len(unique) < 15:
            ax.legend(markerscale=3, fontsize='x-small', loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
donor_id = 151673
print("=== 1. Loading & Assembling Data ===")
adata = sc.read_csv(f"data_full/{donor_id}_counts.csv")
meta = pd.read_csv(f"data_full/{donor_id}_metadata.csv", index_col=0)
adata.obs = meta

# Coords
if 'pxl_row_in_fullres' in adata.obs.columns:
    coords = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
else:
    coords = adata.obs[['row', 'col']].values
adata.obsm['spatial'] = coords.astype(float)

# Ground Truth
gt_key = 'layer_guess_reordered'
if gt_key not in adata.obs.columns: gt_key = 'ManualAnnotation'

# Filter Background
adata = adata[~adata.obs[gt_key].isna()].copy()
adata.var_names_make_unique()
print(f"Data Loaded: {adata.shape}")

# HVG Selection (Corrected Method)
print("\n=== 2. Selecting HVGs (Top 3000) ===")
adata_temp = adata.copy()
sc.pp.filter_genes(adata_temp, min_counts=5)
# sc.pp.normalize_total(adata_temp, target_sum=1e4)
sc.pp.log1p(adata_temp)
sc.pp.highly_variable_genes(adata_temp, n_top_genes=10000, flavor='seurat')
# Robust subsetting by name
hvg_names = adata_temp.var_names[adata_temp.var['highly_variable']]
adata = adata[:, hvg_names].copy()
del adata_temp
print(f"Data Shape after HVG: {adata.shape}")

# Setup Params
Ky = 7 # Brain Layers
Kx = 2000  # Your Budget

print("\n=== 3. Running Models ===")
# A. CiCLODS
adata, selected_genes = run_ciclods_model(adata, Ky, Kx, use_pe=False,  pe_weight=.1)

print(selected_genes)
np.savetxt(f"{donor_id}_ciclods_selected_genes.txt", selected_genes, fmt="%s")
pd.DataFrame({
    'cell_id': adata.obs_names,
    'ciclods_cluster': adata.obs['ciclods'].astype(str).values
}).to_csv(f"{donor_id}_ciclods_cluster_allocations.txt", sep='\t', index=False)

# B. Variance-Matched PCA
adata, n_pcs = run_variance_matched_baseline(adata, selected_genes, Ky)
pd.DataFrame({
    'cell_id': adata.obs_names,
    'matched_baseline_cluster': adata.obs['matched_baseline'].astype(str).values
}).to_csv(f"{donor_id}_pca_cluster_allocations.txt", sep='\t', index=False)

# C. Raw K-Means
adata = run_kmeans_baseline(adata, Ky)
pd.DataFrame({
    'cell_id': adata.obs_names,
    'kmeans_raw_cluster': adata.obs['kmeans_raw'].astype(str).values
}).to_csv(f"{donor_id}_kmeans_cluster_allocations.txt", sep='\t', index=False)

# Calculate ARI
titles = [f"CiCLODS (Kx={Kx})", f"PCA Baseline (k={n_pcs})", "K-Means (No PCA)"]
keys = ['ciclods', 'matched_baseline', 'kmeans_raw']
mask = ~adata.obs[gt_key].isna()

ari1 = adjusted_rand_score(adata.obs.loc[mask, gt_key], adata.obs.loc[mask, 'ciclods'])
ari2 = adjusted_rand_score(adata.obs.loc[mask, gt_key], adata.obs.loc[mask, 'matched_baseline'])
ari3 = adjusted_rand_score(adata.obs.loc[mask, gt_key], adata.obs.loc[mask, 'kmeans_raw'])

titles[0] += f"\nARI={ari1:.3f}"
titles[1] += f"\nARI={ari2:.3f}"
titles[2] += f"\nARI={ari3:.3f}"

# Add Ground Truth for visualization
keys.insert(0, gt_key)
titles.insert(0, "Ground Truth")

print("\n=== 4. Final Results ===")
print(f"CiCLODS:     {ari1:.4f}")
print(f"Matched PCA: {ari2:.4f}")
print(f"Raw K-Means: {ari3:.4f}")

# Plot
plot_ez(adata, keys, titles)
