import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# HELPER: POSITIONAL ENCODING
# ==========================================
def _positional_encoding_2d(coords, total_dims, mode="interleaved"):
    """
    Encodes 2D (x, y) coordinates into a higher-dimensional space using positional encoding.

    Parameters:
    - coords: 2D array (n_samples, 2), input 2D coordinates.
    - total_dims: int, the desired dimensionality of the output encoding.
    - mode: str, the encoding method. Options are "separate", "interleaved", or "norm".

    Returns:
    - encoded_coords: 2D array (n_samples, total_dims), positional-encoded features.
    """
    assert mode in ["separate", "interleaved", 'interaction'], "Mode must be 'separate', 'interleaved', or 'norm'."
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
        encoded_feats = _positional_encoding_2d(spatial_scaled, total_dims=M.shape[1], mode='interleaved')

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
# FUNCTION 1: DATA PREPROCESSING
# ==========================================
def preprocess_spatial_data(adata, n_hvg=2000, target_sum=1e4, spatial_key='spatial'):
    """
    Standardizes preprocessing:
    1. Normalizes counts (Linear).
    2. Logs a COPY to find HVGs (Seurat flavor requires log).
    3. Returns the original Linear data for the selected genes.
    """
    print(f"--- Preprocessing Data (Target HVG: {n_hvg}) ---")
    adata = adata.copy()
    adata.var_names_make_unique()

    # 1. Normalize (Linear Scale)
    # We normalize the main object so the counts are comparable
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # 2. HVG Selection (On Log-Transformed Copy)
    # Seurat flavor expects log-data to calculate dispersion correctly
    adata_log = adata.copy()
    sc.pp.log1p(adata_log)
    sc.pp.highly_variable_genes(adata_log, n_top_genes=n_hvg, flavor='seurat')

    # 3. Filter the ORIGINAL (Linear) Object
    # We grab the names from the log object, but slice the linear object
    hvg_names = adata_log.var_names[adata_log.var['highly_variable']]
    adata = adata[:, hvg_names].copy()

    # 4. Cleanup & Checks
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")

    print(f"    > Data Shape: {adata.shape}")
    print(f"    > Max value check (should be >10 if linear): {adata.X.max():.1f}")

    return adata