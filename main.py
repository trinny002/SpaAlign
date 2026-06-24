import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import argparse
import time
from sklearn.metrics import adjusted_rand_score
from SpaAlign.preprocess import fix_seed
from SpaAlign.preprocess import clr_normalize_each_cell, pca
from SpaAlign.preprocess import construct_neighbor_graph, lsi
from SpaAlign.Train_model import Train
from SpaAlign.utils import clustering

# 3-modality components are only needed for `--data_type Simulation`.
# Some repo snapshots may not include them, so import lazily/defensively.
try:
    from SpaAlign.preprocess_3M import construct_neighbor_graph as construct_neighbor_graph_3M
except ModuleNotFoundError:  # pragma: no cover
    construct_neighbor_graph_3M = None

try:
    from SpaAlign.Train_model_3M import Train_3M
except ModuleNotFoundError:  # pragma: no cover
    Train_3M = None


def _read_gt_labels(gt_path):
    raw_labels = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                raw_labels.append(value)

    mapped = []
    mapping = {}
    next_id = 0
    for value in raw_labels:
        try:
            mapped.append(int(value))
        except ValueError:
            if value not in mapping:
                mapping[value] = next_id
                next_id += 1
            mapped.append(mapping[value])
    return np.asarray(mapped, dtype=np.int64)


def _count_gt_clusters(gt_path: str) -> int:
    """Unique GT classes in file; excludes label -1 (semi-supervised unlabeled)."""
    arr = _read_gt_labels(gt_path)
    valid = arr[arr >= 0]
    if valid.size > 0:
        return int(len(np.unique(valid)))
    return int(len(np.unique(arr)))


def _align_gt_labels_with_barcodes(adata, gt_table_path):
    if not os.path.exists(gt_table_path):
        return None

    df = pd.read_csv(gt_table_path, sep='\t')
    barcode_col = None
    obs_names = set(adata.obs_names.astype(str))
    best_overlap = -1
    for candidate in ['barcode_with_prefix', 'barcode', 'cell', 'cell_id']:
        if candidate not in df.columns:
            continue
        overlap = len(set(df[candidate].astype(str)).intersection(obs_names))
        if overlap > best_overlap:
            best_overlap = overlap
            barcode_col = candidate
    if barcode_col is None:
        return None

    label_col = None
    for candidate in ['label', 'labels', 'gt', 'GT', 'cluster', 'domain']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        return None

    barcode_to_label = {}
    for _, row in df[[barcode_col, label_col]].dropna().iterrows():
        barcode_to_label[str(row[barcode_col])] = row[label_col]

    aligned = np.full(adata.n_obs, -1, dtype=np.int64)
    mapping = {}
    next_id = 0
    matched = 0
    for idx, obs_name in enumerate(adata.obs_names.astype(str)):
        if obs_name not in barcode_to_label:
            continue
        value = barcode_to_label[obs_name]
        try:
            aligned[idx] = int(value)
        except (TypeError, ValueError):
            value = str(value)
            if value not in mapping:
                mapping[value] = next_id
                next_id += 1
            aligned[idx] = mapping[value]
        matched += 1

    if matched == 0:
        return None

    print(
        f"GT alignment from barcode table ({barcode_col}): matched {matched}/{adata.n_obs} samples"
    )
    return aligned


def _save_training_log(log_dict, save_path):
    keys = [
        'loss_total',
        'loss_recon',
        'loss_graph',
        'loss_cl_inst',
        'loss_cl_sup',
        'loss_cl_total',
        'beta_t',
        'epoch_ari',
        'epoch_ari_test',
    ]
    lengths = [len(log_dict.get(key, [])) for key in keys]
    total_epochs = max(lengths) if lengths else 0
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("epoch\t" + "\t".join(keys) + "\n")
        for idx in range(total_epochs):
            row = [str(idx + 1)]
            for key in keys:
                values = log_dict.get(key, [])
                row.append(str(values[idx]) if idx < len(values) else "")
            f.write("\t".join(row) + "\n")
        f.write(f"best_epoch\t{log_dict.get('best_epoch', -1)}\n")
        f.write(f"best_epoch_ari\t{log_dict.get('best_epoch_ari', '')}\n")
        f.write(f"best_epoch_ari_test\t{log_dict.get('best_epoch_ari_test', '')}\n")


def _split_labeled_train_test(labels, train_ratio=1.0, seed=2024, stratified=True):
    labels = np.asarray(labels)
    n = labels.shape[0]
    labeled_idx = np.where(labels >= 0)[0]

    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    if labeled_idx.size == 0:
        return train_mask, test_mask
    if train_ratio >= 1.0:
        train_mask[labeled_idx] = True
        return train_mask, test_mask
    if train_ratio <= 0.0:
        test_mask[labeled_idx] = True
        return train_mask, test_mask

    rng = np.random.RandomState(seed)
    if stratified:
        for cls in np.unique(labels[labeled_idx]):
            cls_idx = np.where(labels == cls)[0]
            cls_idx = cls_idx[labels[cls_idx] >= 0]
            if cls_idx.size == 0:
                continue
            perm = rng.permutation(cls_idx)
            n_train = int(np.floor(cls_idx.size * train_ratio))
            if cls_idx.size > 1:
                n_train = max(1, min(n_train, cls_idx.size - 1))
            else:
                n_train = 1
            train_part = perm[:n_train]
            test_part = perm[n_train:]
            train_mask[train_part] = True
            test_mask[test_part] = True
    else:
        perm = rng.permutation(labeled_idx)
        n_train = int(np.floor(labeled_idx.size * train_ratio))
        if labeled_idx.size > 1:
            n_train = max(1, min(n_train, labeled_idx.size - 1))
        else:
            n_train = 1
        train_mask[perm[:n_train]] = True
        test_mask[perm[n_train:]] = True

    return train_mask, test_mask


def _save_label_split_table(adata, labels_eval, train_mask, test_mask, save_path):
    split = np.full(adata.n_obs, 'unlabeled', dtype=object)
    split[train_mask] = 'train'
    split[test_mask] = 'test'
    df = pd.DataFrame(
        {
            'obs_name': adata.obs_names.astype(str),
            'label': labels_eval,
            'split': split,
        }
    )
    df.to_csv(save_path, sep='\t', index=False)


def _align_modalities(adata_omics1, adata_omics2):
    shared = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    if len(shared) == 0:
        raise ValueError("Two modalities have no overlapping cells; cannot align")
    ordered = adata_omics1.obs_names[adata_omics1.obs_names.isin(shared)]
    adata_omics1 = adata_omics1[ordered].copy()
    adata_omics2 = adata_omics2[ordered].copy()
    return adata_omics1, adata_omics2

def _is_integer_matrix(adata, max_check=1000):
    x = adata.X
    if sp.issparse(x):
        data = x.data
    else:
        data = np.asarray(x).ravel()
    if data.size == 0:
        return False
    if data.size > max_check:
        data = data[:max_check]
    return np.allclose(data, np.round(data))

def _highly_variable_genes_counts(adata, n_top_genes):
    counts_layer = None
    if 'counts' in adata.layers:
        counts_layer = 'counts'
    elif 'count' in adata.layers:
        counts_layer = 'count'

    if counts_layer is not None:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            layer=counts_layer,
        )
        return
    if adata.raw is not None:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            use_raw=True,
        )
        return
    if _is_integer_matrix(adata):
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
        )
        return
    raise ValueError(
        "HVG (seurat_v3) requires raw counts. Provide counts in adata.layers['counts'] or adata.raw."
    )

def _replace_nan_inplace(adata):
    x = adata.X
    if sp.issparse(x):
        data = x.data
        if data.size:
            data[np.isnan(data)] = 0.0
    else:
        if np.isnan(x).any():
            adata.X = np.nan_to_num(x, nan=0.0)

def _ensure_counts_in_X(adata):
    for layer_name in ('counts', 'count'):
        if layer_name in adata.layers:
            adata.X = adata.layers[layer_name].copy()
            return True
    return False

def _assert_no_negative_counts(adata, context):
    x = adata.X
    if sp.issparse(x):
        has_negative = x.data.size and (x.data < 0).any()
    else:
        has_negative = np.asarray(x).min() < 0
    if has_negative:
        raise ValueError(
            f"{context} expects raw counts, but adata.X has negative values. "
            "Provide counts in adata.layers['counts'] or adata.layers['count']."
        )

def main(args):
    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if getattr(args, "infer_n_clusters_from_gt", False):
        src = (getattr(args, "n_clusters_gt_path", None) or "").strip() or (args.gt_path or "").strip()
        if not src or not os.path.isfile(src):
            raise ValueError(
                "--infer_n_clusters_from_gt requires an existing file: set --n_clusters_gt_path "
                "or --gt_path to GT_labels.txt"
            )
        args.n_clusters = _count_gt_clusters(src)
        print(f"Inferred n_clusters={args.n_clusters} from unique labels in {src} (labels < 0 ignored).")
    if args.n_clusters is None:
        raise ValueError("Provide --n_clusters or use --infer_n_clusters_from_gt with --gt_path.")

    if getattr(args, "sync_init_k_with_n_clusters", False):
        args.init_k = int(args.n_clusters)
        print(f"sync_init_k_with_n_clusters: init_k set to {args.init_k}")

   # read data
    if args.data_type in ['10x', 'SPOTS', 'Stereo-CITE-seq']:
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_RNA.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ADT.h5ad'))
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 's1_adata_rna.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 's1_adata_atac.h5ad'))
    elif args.data_type == 'Simulation':
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_RNA.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ADT.h5ad'))
        adata_omics3 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ATAC.h5ad'))
    else:
        raise ValueError(
            f"Unknown --data_type {args.data_type!r}. "
            "Use 10x | SPOTS | Stereo-CITE-seq | Spatial-epigenome-transcriptome | Simulation. "
            "Note: only Simulation uses 3 modalities (RNA+ADT+ATAC) and Train_3M; all others use 2 modalities and Train."
        )

    if args.data_type == 'Simulation':
        print("[SpaAlign] 3-modality pipeline: RNA + ADT + ATAC -> construct_neighbor_graph_3M + Train_3M")
    else:
        print(f"[SpaAlign] 2-modality pipeline: data_type={args.data_type!r} -> construct_neighbor_graph + Train")

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    if args.data_type in ['10x', 'SPOTS', 'Stereo-CITE-seq']:
        adata_omics1, adata_omics2 = _align_modalities(adata_omics1, adata_omics2)
    if args.data_type == 'Simulation':
        adata_omics3.var_names_make_unique()

        # Fix random seed

    random_seed = args.seed
    fix_seed(random_seed)

    # Preprocess
    if args.data_type == '10x':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        _ensure_counts_in_X(adata_omics1)
        _replace_nan_inplace(adata_omics1)
        _assert_no_negative_counts(adata_omics1, "RNA preprocessing")
        _highly_variable_genes_counts(adata_omics1, n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        _replace_nan_inplace(adata_omics1_high)
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        _ensure_counts_in_X(adata_omics2)
        _replace_nan_inplace(adata_omics2)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        _replace_nan_inplace(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        # RNA
        ##sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=200)
        _ensure_counts_in_X(adata_omics1)
        _replace_nan_inplace(adata_omics1)
        _assert_no_negative_counts(adata_omics1, "RNA preprocessing")
        _highly_variable_genes_counts(adata_omics1, n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        _replace_nan_inplace(adata_omics1_high)
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)
        # ATAC
        adata_omics2 = adata_omics2[
            adata_omics1.obs_names].copy()  # .obsm['X_lsi'] represents the dimension reduced feature
        if 'X_lsi' not in adata_omics2.obsm.keys():
            _ensure_counts_in_X(adata_omics2)
            _replace_nan_inplace(adata_omics2)
            _assert_no_negative_counts(adata_omics2, "ATAC preprocessing")
            _highly_variable_genes_counts(adata_omics2, n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'SPOTS':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        _ensure_counts_in_X(adata_omics1)
        _replace_nan_inplace(adata_omics1)
        _assert_no_negative_counts(adata_omics1, "RNA preprocessing")
        _highly_variable_genes_counts(adata_omics1, n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        _replace_nan_inplace(adata_omics1_high)
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        _ensure_counts_in_X(adata_omics2)
        _replace_nan_inplace(adata_omics2)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        _replace_nan_inplace(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Stereo-CITE-seq':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=80)
        sc.pp.filter_genes(adata_omics2, min_cells=50)
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        _ensure_counts_in_X(adata_omics1)
        _replace_nan_inplace(adata_omics1)
        _assert_no_negative_counts(adata_omics1, "RNA preprocessing")
        _highly_variable_genes_counts(adata_omics1, n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        _replace_nan_inplace(adata_omics1_high)
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        _ensure_counts_in_X(adata_omics2)
        _replace_nan_inplace(adata_omics2)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        _replace_nan_inplace(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Simulation':
        n_protein = adata_omics2.n_vars
        _ensure_counts_in_X(adata_omics1)
        _replace_nan_inplace(adata_omics1)
        _assert_no_negative_counts(adata_omics1, "RNA preprocessing")
        _highly_variable_genes_counts(adata_omics1, n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        _replace_nan_inplace(adata_omics1_high)
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_protein)
        # Protein
        _ensure_counts_in_X(adata_omics2)
        _replace_nan_inplace(adata_omics2)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        _replace_nan_inplace(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_protein)
        # ATAC
        _ensure_counts_in_X(adata_omics3)
        _replace_nan_inplace(adata_omics3)
        _assert_no_negative_counts(adata_omics3, "ATAC preprocessing")
        _highly_variable_genes_counts(adata_omics3, n_top_genes=3000)
        lsi(adata_omics3, use_highly_variable=False, n_components=n_protein + 1)
        adata_omics3.obsm['feat'] = adata_omics3.obsm['X_lsi'].copy()

        
        if construct_neighbor_graph_3M is None or Train_3M is None:
            raise ModuleNotFoundError(
                "3-modality modules are missing: expected SpaAlign.preprocess_3M/Train_model_3M "
                "to run with --data_type Simulation. For SPOTS/10x/etc, keep --data_type non-Simulation."
            )
        data = construct_neighbor_graph_3M(adata_omics1, adata_omics2, adata_omics3)

    if getattr(args, 'gt_path', None):
        gt_labels = None
        if os.path.exists(args.gt_path):
            raw_gt_labels = _read_gt_labels(args.gt_path)
            if raw_gt_labels.shape[0] == adata_omics1.n_obs:
                gt_labels = raw_gt_labels
            else:
                gt_table_path = getattr(args, 'gt_table_path', '')
                if not gt_table_path:
                    default_table = os.path.join(os.path.dirname(args.gt_path), 'GT_labels_with_barcodes.tsv')
                    gt_table_path = default_table if os.path.exists(default_table) else ''
                if gt_table_path:
                    gt_labels = _align_gt_labels_with_barcodes(adata_omics1, gt_table_path)
                if gt_labels is None:
                    print(
                        f"Warning: gt label length mismatch ({raw_gt_labels.shape[0]} vs {adata_omics1.n_obs}), "
                        "and barcode alignment unavailable; supervised contrastive disabled."
                    )
        else:
            print(f"Warning: gt_path not found: {args.gt_path}; supervised contrastive disabled.")

        if gt_labels is not None:
            gt_labels_eval = gt_labels.copy()
            gt_labels_train = gt_labels.copy()

            train_mask, test_mask = _split_labeled_train_test(
                gt_labels_eval,
                train_ratio=args.label_train_ratio,
                seed=args.label_split_seed,
                stratified=(not args.disable_stratified_split),
            )
            if args.label_train_ratio < 1.0:
                gt_labels_train[test_mask] = -1
                print(
                    f"Label split: train={int(train_mask.sum())}, test={int(test_mask.sum())}, unlabeled={int((gt_labels_eval < 0).sum())}"
                )

            data['gt_labels'] = gt_labels_train
            data['gt_labels_eval'] = gt_labels_eval
            data['gt_train_mask'] = train_mask
            data['gt_test_mask'] = test_mask

            if args.save_label_split:
                split_path = args.txt_out_path.replace('.txt', '_label_split.tsv')
                _save_label_split_table(adata_omics1, gt_labels_eval, train_mask, test_mask, split_path)

    
    # define model
    if args.data_type == 'Simulation':
        if Train_3M is None:
            raise ModuleNotFoundError(
                "SpaAlign.Train_model_3M is missing; cannot run with --data_type Simulation."
            )
        model = Train_3M(
            data,
            datatype=args.data_type,
            device=device,
            Arg=args,
            use_dynamic_graph=args.use_dynamic_graph,
        )
    else:
        model = Train(
            data,
            datatype=args.data_type,
            device=device,
            random_seed=random_seed,
            Arg=args,
            use_alignment=args.use_alignment,
            use_dual_stream=args.use_dual_stream,
            use_dynamic_graph=args.use_dynamic_graph,
        )

    start_time = time.time()

    # train model
    output = model.train()

    if getattr(args, 'save_train_log', False) and 'training_log' in output:
        train_log_path = args.txt_out_path.replace('.txt', '_train_log.tsv')
        _save_training_log(output['training_log'], train_log_path)

    end_time = time.time()

    print("Training time: ", end_time - start_time)
    # torch.save(model.model.state_dict(), 'model_weights/SpaAlign_' + args.data_type + '.pth')

    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['SpaAlign'] = output['SpaAlign'].copy()

    if np.isnan(adata.obsm['SpaAlign']).any():
        print("Warning: SpaAlign embedding has NaN; replacing with 0 for clustering.")
        adata.obsm['SpaAlign'] = np.nan_to_num(adata.obsm['SpaAlign'], nan=0.0)

    clustering(
        adata,
        key='SpaAlign',
        add_key='SpaAlign',
        n_clusters=args.n_clusters,
        method=args.cluster_method,
        use_pca=True,
        n_comps=args.cluster_n_comps,
        random_seed=random_seed,
        mclust_model=args.mclust_model,
        start=args.cluster_res_start,
        end=args.cluster_res_end,
        increment=args.cluster_res_step,
        graph_n_neighbors=args.cluster_graph_n_neighbors,
    )

    label = adata.obs['SpaAlign']

    if args.data_type == 'Simulation':
        ids = label.index.astype(str).str[:4]
        int_list = [int(num_str) for num_str in ids]
        list = [-1 for i in range(len(int_list))]
        for i in range(len(int_list)):
            list[int_list[i]] = label[i]
        spot_size = 60
    else:
        list = label.tolist()
        spot_size = 20

    # Save results
    output_file = args.txt_out_path
    with open(output_file, 'w') as f:
        for num in list:
            f.write(f"{num}\n")

    if getattr(args, 'save_h5ad_path', ''):
        h5ad_dir = os.path.dirname(args.save_h5ad_path)
        if h5ad_dir:
            os.makedirs(h5ad_dir, exist_ok=True)
        adata.write_h5ad(args.save_h5ad_path)
        print(f"Saved AnnData with obs['SpaAlign'] to: {args.save_h5ad_path}")

    gt_labels_eval = data.get('gt_labels_eval', None)
    gt_train_mask = data.get('gt_train_mask', None)
    gt_test_mask = data.get('gt_test_mask', None)
    if gt_labels_eval is not None and len(gt_labels_eval) == len(list):
        pred_codes = pd.factorize(np.asarray(list).astype(str))[0]
        gt_eval = np.asarray(gt_labels_eval)
        labeled_mask = gt_eval >= 0

        semi_eval_path = args.txt_out_path.replace('.txt', '_semi_eval.txt')
        with open(semi_eval_path, 'w', encoding='utf-8') as sf:
            sf.write(f"n_total\t{len(gt_eval)}\n")
            sf.write(f"n_labeled\t{int(labeled_mask.sum())}\n")
            if labeled_mask.sum() > 1:
                ari_all_labeled = adjusted_rand_score(gt_eval[labeled_mask], pred_codes[labeled_mask])
                sf.write(f"ARI_labeled\t{ari_all_labeled:.6f}\n")
            else:
                sf.write("ARI_labeled\tNA\n")

            if gt_train_mask is not None:
                train_eval_mask = np.asarray(gt_train_mask) & labeled_mask
                sf.write(f"n_train_labeled\t{int(train_eval_mask.sum())}\n")
                if train_eval_mask.sum() > 1:
                    ari_train = adjusted_rand_score(gt_eval[train_eval_mask], pred_codes[train_eval_mask])
                    sf.write(f"ARI_train_labeled\t{ari_train:.6f}\n")
                else:
                    sf.write("ARI_train_labeled\tNA\n")

            if gt_test_mask is not None:
                test_eval_mask = np.asarray(gt_test_mask) & labeled_mask
                sf.write(f"n_test_labeled\t{int(test_eval_mask.sum())}\n")
                if test_eval_mask.sum() > 1:
                    ari_test = adjusted_rand_score(gt_eval[test_eval_mask], pred_codes[test_eval_mask])
                    sf.write(f"ARI_test_labeled\t{ari_test:.6f}\n")
                else:
                    sf.write("ARI_test_labeled\tNA\n")

        pred_labeled_path = args.txt_out_path.replace('.txt', '_labeled_pred.txt')
        gt_labeled_path = args.txt_out_path.replace('.txt', '_labeled_gt.txt')
        with open(pred_labeled_path, 'w', encoding='utf-8') as pf, open(gt_labeled_path, 'w', encoding='utf-8') as gf:
            for pred_val, gt_val, keep in zip(pred_codes, gt_eval, labeled_mask):
                if keep:
                    pf.write(f"{int(pred_val)}\n")
                    gf.write(f"{int(gt_val)}\n")

    # visualization
    if args.skip_vis:
        print("Info: --skip_vis enabled, skip UMAP/spatial plotting and SVG export.")
        return

    spatial_basis = None
    if 'spatial' in adata.obsm:
        spatial_basis = 'spatial'
    elif 'X_spatial' in adata.obsm:
        spatial_basis = 'X_spatial'

    if spatial_basis is None:
        obs_x_candidates = ['array_col', 'x', 'X', 'coord_x', 'px', 'imagecol', 'image_col', 'col']
        obs_y_candidates = ['array_row', 'y', 'Y', 'coord_y', 'py', 'imagerow', 'image_row', 'row']
        for cx in obs_x_candidates:
            for cy in obs_y_candidates:
                if cx in adata.obs and cy in adata.obs:
                    adata.obsm['spatial'] = np.vstack([
                        adata.obs[cx].astype(float).values,
                        adata.obs[cy].astype(float).values,
                    ]).T
                    spatial_basis = 'spatial'
                    if cx == 'array_col' and cy == 'array_row':
                        print("Info: using adata.obs['array_col'], adata.obs['array_row'] as spatial basis for plotting.")
                    else:
                        print(f"Warning: using adata.obs['{cx}'], adata.obs['{cy}'] as spatial basis for plotting.")
                    break
            if spatial_basis is not None:
                break

    if spatial_basis is None:
        for k in adata.obsm_keys():
            arr = np.asarray(adata.obsm[k])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                adata.obsm['spatial'] = arr[:, :2].copy()
                spatial_basis = 'spatial'
                print(f"Warning: using adata.obsm['{k}'] as spatial basis for plotting.")
                break

    if spatial_basis == 'spatial' and args.data_type == 'Stereo-CITE-seq':
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
    elif spatial_basis == 'spatial' and args.data_type == 'SPOTS':
        # flip tissue image
        adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

    import matplotlib.pyplot as plt
    fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
    sc.pp.neighbors(adata, use_rep='SpaAlign', n_neighbors=500)
    sc.tl.umap(adata)

    sc.pl.umap(
        adata,
        color='SpaAlign',
        ax=ax_list[0],
        title='SpaAlign',
        s=spot_size,
        add_outline=True,
        outline_color=('black', 'white'),
        outline_width=(0.25, 0.05),
        show=False,
    )
    if spatial_basis is not None:
        sc.pl.embedding(
            adata,
            basis=spatial_basis,
            color='SpaAlign',
            ax=ax_list[1],
            title='SpaAlign',
            s=spot_size,
            add_outline=True,
            outline_color=('black', 'white'),
            outline_width=(0.25, 0.05),
            show=False,
        )
    else:
        ax_list[1].axis('off')
        ax_list[1].set_title('SpaAlign (no spatial basis)')
        print("Warning: no spatial coordinates found; skip spatial embedding plot.")

    plt.tight_layout(w_pad=0.3)
    plt.savefig(args.vis_out_path)

    # 额外保存为 SVG，便于论文级矢量图展示（默认输出到 results/clustergraph）
    svg_dir = args.cluster_svg_dir
    os.makedirs(svg_dir, exist_ok=True)
    svg_name = os.path.splitext(os.path.basename(args.txt_out_path))[0] + '.svg'
    svg_path = os.path.join(svg_dir, svg_name)
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved cluster SVG to: {svg_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--file_fold', type=str,
                        help='Path to data folder')
    parser.add_argument('--data_type', type=str,
                        choices=['10x', 'Spatial-epigenome-transcriptome', 'SPOTS', 'Stereo-CITE-seq', 'Simulation'],
                        help='data_type')
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=None,
        help='Number of clusters for mclust/leiden/louvain. Omit if using --infer_n_clusters_from_gt.',
    )
    parser.add_argument(
        '--infer_n_clusters_from_gt',
        action='store_true',
        help='Set n_clusters from unique labels in --n_clusters_gt_path if set, else in --gt_path (excludes -1).',
    )
    parser.add_argument(
        '--n_clusters_gt_path',
        type=str,
        default='',
        help='Optional GT file used only to count classes with --infer_n_clusters_from_gt (no supervised loss by itself).',
    )
    parser.add_argument(
        '--sync_init_k_with_n_clusters',
        action='store_true',
        help='After n_clusters is known, set --init_k to the same value (optional convenience).',
    )

    parser.add_argument('--init_k', type=int, default=10, help='init k')
    parser.add_argument('--KNN_k', type=int, default=20, help='KNN_k')
    parser.add_argument('--alpha', type=float, default=0.9, help='init k')
    parser.add_argument('--cl_weight', type=float, default=1, help='weight')
    parser.add_argument('--RNA_weight', type=float, default=5, help='weight')
    parser.add_argument('--ADT_weight', type=float, default=5, help='weight')
    parser.add_argument('--contrastive_weight', type=float, default=0.2, help='contrastive loss weight')
    parser.add_argument('--tau', type=float, default=2, help='temperature')
    parser.add_argument('--tau_inst', type=float, default=0.1, help='instance-level contrastive temperature')
    parser.add_argument('--tau_sup', type=float, default=0.1, help='supervised contrastive temperature')
    parser.add_argument('--cl_alpha', type=float, default=1.0, help='instance-level contrastive coefficient')
    parser.add_argument('--cl_beta', type=float, default=0.5, help='supervised contrastive coefficient')
    parser.add_argument('--warmup_ratio', type=float, default=0.3, help='warmup ratio for supervised contrastive coefficient')
    parser.add_argument('--hard_negative', action='store_true', help='Enable hard negative mining for supervised contrastive')
    parser.add_argument('--hard_k_hard', type=int, default=2, help='number of hard negatives per anchor')
    parser.add_argument('--hard_k_easy', type=int, default=2, help='number of easy negatives per anchor')
    parser.add_argument('--gt_path', type=str, default='', help='GT labels path for supervised contrastive training')
    parser.add_argument('--gt_table_path', type=str, default='', help='Optional GT table (tsv) with barcode and label columns')
    parser.add_argument('--label_train_ratio', type=float, default=1.0, help='Ratio of labeled samples used as supervised train labels')
    parser.add_argument('--label_split_seed', type=int, default=2024, help='Random seed for label train/test split')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed for training, clustering, and global RNG state')
    parser.add_argument('--epochs', type=int, default=None, help='Optional override for training epochs')
    parser.add_argument('--learning_rate', type=float, default=None, help='Optional override for optimizer learning rate')
    parser.add_argument('--disable_stratified_split', action='store_true', help='Disable class-stratified label split')
    parser.add_argument('--save_label_split', action='store_true', help='Save label split table as *_label_split.tsv')
    parser.add_argument('--track_epoch_ari', action='store_true', help='Track per-epoch ARI using GT labels')
    parser.add_argument('--ari_eval_interval', type=int, default=5, help='evaluate per-epoch ARI interval')
    parser.add_argument('--save_train_log', action='store_true', help='save per-epoch training log as tsv')
    parser.add_argument('--vis_out_path', type=str, default='results/HLN.png', help='vis_out_path')
    parser.add_argument('--txt_out_path', type=str, default='results/HLN.txt', help='txt_out_path')
    parser.add_argument('--save_h5ad_path', type=str, default='', help='Optional path to save AnnData with obs[\'SpaAlign\']')
    parser.add_argument('--cluster_svg_dir', type=str, default='results/clustergraph', help='Directory to save SVG cluster visualization')
    parser.add_argument(
        '--cluster_method',
        type=str,
        default='mclust',
        choices=['mclust', 'leiden', 'louvain'],
        help="Post-hoc clustering on obsm['SpaAlign'] (after optional PCA; see SpaAlign.utils.clustering)",
    )
    parser.add_argument('--cluster_n_comps', type=int, default=20, help='PCA components before mclust / graph clustering when use_pca=True')
    parser.add_argument(
        '--mclust_model',
        type=str,
        default='EEE',
        help="R mclust modelNames (e.g. EEE, VVV, EII, VII); see Mclust documentation",
    )
    parser.add_argument('--cluster_res_start', type=float, default=0.1, help='Leiden/Louvain resolution search start')
    parser.add_argument('--cluster_res_end', type=float, default=3.0, help='Leiden/Louvain resolution search end')
    parser.add_argument('--cluster_res_step', type=float, default=0.01, help='Leiden/Louvain resolution search step')
    parser.add_argument(
        '--cluster_graph_n_neighbors',
        type=int,
        default=50,
        help='n_neighbors when building the scanpy graph for Leiden/Louvain resolution search',
    )
    parser.add_argument('--skip_vis', action='store_true', help='Skip UMAP/spatial plotting and SVG export')
    parser.add_argument('--disable_alignment', action='store_true', help='Disable RNA-query alignment module')
    parser.add_argument('--disable_dual_stream', action='store_true', help='Disable dual-stream alignment module')
    parser.add_argument('--disable_spatial', action='store_true', help='Disable spatial stream inside dual-stream alignment')
    parser.add_argument('--disable_semantic', action='store_true', help='Disable semantic stream inside dual-stream alignment')
    parser.add_argument('--disable_contrastive', action='store_true', help='Disable contrastive learning loss')
    parser.add_argument('--disable_dynamic_graph', action='store_true', help='Disable learnable graph (use KNN adjacency)')
    args = parser.parse_args()
    args.use_alignment = not args.disable_alignment
    args.use_dual_stream = not args.disable_dual_stream
    args.use_spatial = not args.disable_spatial
    args.use_semantic = not args.disable_semantic
    args.use_contrastive = not args.disable_contrastive
    args.use_dynamic_graph = not args.disable_dynamic_graph
    main(args)
