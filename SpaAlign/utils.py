"""utils.py

工具函数与聚类封装。包含：
- `mclust_R`：通过 `rpy2` 调用 R 的 `mclust` 包进行高斯混合模型聚类（需要系统安装 R 与 mclust）。
- `clustering`：在 `mclust`、`leiden`、`louvain` 三者中选择聚类方法并把结果写入 `adata.obs`。

注意：`mclust_R` 依赖系统层面的 R 与 R 包 `mclust`，在无 R 环境时会失败；如果无法依赖 R，建议使用 `method='leiden'` 或 `method='louvain'`。
"""

import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from .preprocess import pca
import matplotlib.pyplot as plt

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    data = np.asarray(adata.obsm[used_obsm])
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    nrow, ncol = data.shape
    if ncol == 0:
        raise RuntimeError("mclust failed: input has 0 columns after preprocessing.")

    data_f = np.asfortranarray(data, dtype=float)
    r_matrix = robjects.r.matrix(
        robjects.FloatVector(data_f.ravel(order='F')),
        nrow=nrow,
        ncol=ncol,
    )
    rmclust_safe = robjects.r(
        'function(x, G, modelNames) { dimnames(x) <- list(NULL, NULL); Mclust(x, G=G, modelNames=modelNames) }'
    )

    res = rmclust_safe(r_matrix, int(num_cluster), str(modelNames))
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(
    adata,
    n_clusters=7,
    key='emb',
    add_key='SpatialGlue',
    method='mclust',
    start=0.1,
    end=3.0,
    increment=0.01,
    use_pca=False,
    n_comps=20,
    random_seed=2024,
    mclust_model='EEE',
    graph_n_neighbors=50,
):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
        print(
            f"mclust input: {key + '_pca'} shape={adata.obsm[key + '_pca'].shape}, "
            f"n_clusters={n_clusters}"
        )
    
    if method == 'mclust':
        if use_pca:
            adata = mclust_R(
                adata,
                used_obsm=key + '_pca',
                num_cluster=n_clusters,
                random_seed=random_seed,
                modelNames=mclust_model,
            )
        else:
            adata = mclust_R(
                adata,
                used_obsm=key,
                num_cluster=n_clusters,
                random_seed=random_seed,
                modelNames=mclust_model,
            )
        adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
        if use_pca:
            res = search_res(
                adata,
                n_clusters,
                use_rep=key + '_pca',
                method=method,
                start=start,
                end=end,
                increment=increment,
                n_neighbors=graph_n_neighbors,
            )
        else:
            res = search_res(
                adata,
                n_clusters,
                use_rep=key,
                method=method,
                start=start,
                end=end,
                increment=increment,
                n_neighbors=graph_n_neighbors,
            )
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca:
          res = search_res(
              adata,
              n_clusters,
              use_rep=key + '_pca',
              method=method,
              start=start,
              end=end,
              increment=increment,
              n_neighbors=graph_n_neighbors,
          )
       else:
          res = search_res(
              adata,
              n_clusters,
              use_rep=key,
              method=method,
              start=start,
              end=end,
              increment=increment,
              n_neighbors=graph_n_neighbors,
          )
       sc.tl.louvain(adata, random_state=random_seed, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']
       
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01, n_neighbors=50):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=int(n_neighbors), use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

