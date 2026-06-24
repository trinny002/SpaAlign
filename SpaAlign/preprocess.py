"""preprocess.py

预处理模块：负责数据标准化、PCA/LSI 降维、基于坐标或表达构建邻接（空间图与特征图），以及把邻接转换为 PyTorch 可用的稀疏矩阵格式。

重要接口：
- `construct_neighbor_graph`: 读取 AnnData 中的 `obsm['spatial']` 和 `obsm['feat']`，构建空间邻接与特征邻接。
- `adjacent_matrix_preprocessing`: 将 pandas 风格的邻接表或 scipy 矩阵转换为归一化的 torch sparse tensor，供模型使用。
- `fix_seed`: 固定随机数种子以便可复现（包括 CUDA 相关的种子设置）。
"""

import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

'''
---------------------
author: Yahui Long https://github.com/JinmiaoChenLab/SpatialGlue
e-mail: chen_jinmiao@bii.a-star.edu.sg
AGPL-3.0 LICENSE
---------------------
'''
    
def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3, Arg=None):
    """
    Construct neighbor graphs, including feature graph and spatial graph. 
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
       n_neighbors=6 
    # omics1 / omics2: robustly obtain spatial coordinates from common locations
    def _get_spatial_coords(adata, name="adata"):
        # try common obsm keys
        prefer_keys = ['spatial', 'spatial_coords', 'coords', 'spatial0', 'X_spatial']
        for k in prefer_keys:
            if k in adata.obsm:
                val = adata.obsm[k]
                try:
                    arr = np.asarray(val)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        return arr
                except Exception:
                    pass
        # try any obsm entry that looks like 2D coordinates
        for k in adata.obsm_keys():
            val = adata.obsm[k]
            try:
                arr = np.asarray(val)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr
            except Exception:
                continue
        # try obs columns (common names)
        obs_x_candidates = ['x', 'X', 'coord_x', 'px', 'imagecol', 'image_col', 'col']
        obs_y_candidates = ['y', 'Y', 'coord_y', 'py', 'imagerow', 'image_row', 'row']
        for cx in obs_x_candidates:
            for cy in obs_y_candidates:
                if cx in adata.obs and cy in adata.obs:
                    try:
                        xs = adata.obs[cx].astype(float).values
                        ys = adata.obs[cy].astype(float).values
                        coords = np.vstack([xs, ys]).T
                        return coords
                    except Exception:
                        continue
        raise KeyError(f"Spatial coordinates not found for {name}. Available obsm keys: {list(adata.obsm.keys())}.\nPlease ensure spatial coordinates are stored in `adata.obsm['spatial']` or provide columns in `adata.obs` named e.g. ('x','y').")

    # omics1
    cell_position_omics1 = _get_spatial_coords(adata_omics1, name='omics1')
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    cell_position_omics2 = _get_spatial_coords(adata_omics2, name='omics2')
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2
    
    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2, k=Arg.KNN_k)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    
    return data

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)

    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode= "distance", metric="correlation", include_self=False):
    
    """Constructing feature neighbor graph according to expresss profiles"""
    
    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    """Constructing spatial neighbor graph according to spatial coordinates."""
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    
    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)
    
    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()
    
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)
    
    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)
    
    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())
    
    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1>1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2>1, 1, adj_feature_omics2)
    
    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1) # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)
    
    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }
    
    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    # avoid divide-by-zero: add safe-guard for zero column sums
    if scipy.sparse.issparse(X):
        col_sum = np.array(X.sum(axis=0)).ravel()
    else:
        col_sum = np.sum(X, axis=0)
    # replace zeros with 1 to avoid divide-by-zero; those columns have idf = n_samples
    col_sum_safe = col_sum.copy()
    col_sum_safe[col_sum_safe == 0] = 1.0
    idf = float(X.shape[0]) / col_sum_safe

    if scipy.sparse.issparse(X):
        row_sum = np.array(X.sum(axis=1)).ravel()
        # avoid dividing by zero rows
        row_sum_safe = row_sum.copy()
        row_sum_safe[row_sum_safe == 0] = 1.0
        inv_row = 1.0 / row_sum_safe
        tf = X.multiply(inv_row[:, None])
        return tf.multiply(idf)
    else:
        row_sum = X.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        tf = X / row_sum
        return tf * idf
    
def fix_seed(seed):
    #seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    
