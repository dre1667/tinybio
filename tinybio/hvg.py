"""Highly-variable gene selection.

v0.1 uses a simplified "top-N by log-variance" ranking. Scanpy's
``seurat_v3`` method bins genes by mean before ranking, which better
handles mean-variance confounding in raw counts; that's a future
improvement. For log1p-normalized data the simpler variant is close
enough for PCA input and avoids a dependency on CPU-side binning.
"""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor


def select_highly_variable(X: Tensor, n_top: int = 2000) -> np.ndarray:
    """Return indices of the top ``n_top`` columns (genes) by variance.

    ``X`` is expected to be log1p-CPM-normalized, shape ``(n_cells, n_genes)``.
    Computes per-column variance on the device, transfers the 1-D
    ``(n_genes,)`` vector to CPU for argsort, and returns a sorted numpy
    ``int64`` array of column indices.
    """
    n_genes = int(X.shape[1])
    if n_top >= n_genes:
        return np.arange(n_genes, dtype=np.int64)
    gene_var = X.var(axis=0).realize().numpy()
    top = np.argpartition(gene_var, -n_top)[-n_top:]
    return np.sort(top).astype(np.int64)
