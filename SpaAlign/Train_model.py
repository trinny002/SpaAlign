"""Train_model.py

训练器模块：封装了训练循环、损失计算与模型评估接口。

主要类：
- Train: 构建 `Encoder_overall`、参数化邻接矩阵（`Parametered_Graph`）、执行训练循环并在训练结束后返回嵌入表示。
- Parametered_Graph: 将输入邻接矩阵作为可学习参数，返回归一化的对称邻接矩阵。

约定：
- 默认 `dim_output=64`（潜在表示维度 L=64）。
"""

import torch
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from .model import Encoder_overall
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score
from .preprocess import adjacent_matrix_preprocessing


def info_nce_inst(q, k, tau=0.07, eps=1e-12):
    q = F.normalize(q, p=2, dim=1)
    k = F.normalize(k, p=2, dim=1)
    tau = max(float(tau), eps)
    labels = torch.arange(q.size(0), device=q.device)

    logits_qk = torch.matmul(q, k.t()) / tau
    logits_qk = logits_qk - logits_qk.max(dim=1, keepdim=True).values
    loss_qk = F.cross_entropy(logits_qk, labels)

    logits_kq = torch.matmul(k, q.t()) / tau
    logits_kq = logits_kq - logits_kq.max(dim=1, keepdim=True).values
    loss_kq = F.cross_entropy(logits_kq, labels)
    return 0.5 * (loss_qk + loss_kq)


def hard_negative_sampling(sim, y, k_hard=2, k_easy=2):
    n = sim.size(0)
    device = sim.device
    neg_mask = torch.zeros((n, n), dtype=torch.bool, device=device)

    for i in range(n):
        if y[i] < 0:
            continue
        diff_idx = torch.where((y != y[i]) & (y >= 0))[0]
        if diff_idx.numel() == 0:
            continue

        scores = sim[i, diff_idx]
        chosen_idx = []
        hard_num = min(int(k_hard), diff_idx.numel())
        easy_num = min(int(k_easy), diff_idx.numel())

        if hard_num > 0:
            hard_pos = torch.topk(scores, k=hard_num, largest=True).indices
            chosen_idx.append(diff_idx[hard_pos])
        if easy_num > 0:
            easy_pos = torch.topk(scores, k=easy_num, largest=False).indices
            chosen_idx.append(diff_idx[easy_pos])

        if chosen_idx:
            chosen = torch.unique(torch.cat(chosen_idx))
            neg_mask[i, chosen] = True

    return neg_mask


def sup_infonce(z, y, tau=0.1, eps=1e-12, use_hard_negative=False, k_hard=2, k_easy=2):
    z = F.normalize(z, p=2, dim=1)
    tau = max(float(tau), eps)

    sim = torch.matmul(z, z.t()) / tau
    sim = sim - sim.max(dim=1, keepdim=True).values

    n = z.size(0)
    eye_mask = torch.eye(n, device=z.device, dtype=torch.bool)
    labeled_mask = y >= 0
    if labeled_mask.sum() <= 1:
        return z.new_tensor(0.0)
    same_label = y.view(-1, 1).eq(y.view(1, -1)) & labeled_mask.view(-1, 1) & labeled_mask.view(1, -1)

    pos_mask = same_label & (~eye_mask)
    valid_mask = (pos_mask.sum(dim=1) > 0) & labeled_mask
    if not valid_mask.any():
        return z.new_tensor(0.0)

    if use_hard_negative:
        neg_mask = hard_negative_sampling(sim.detach(), y, k_hard=k_hard, k_easy=k_easy)
        denom_mask = pos_mask | neg_mask
    else:
        denom_mask = (~eye_mask) & labeled_mask.view(1, -1)

    exp_logits = torch.exp(sim) * denom_mask.float()
    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)
    log_prob = sim - torch.log(denom)

    pos_count = pos_mask.sum(dim=1).clamp_min(1).float()
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count
    loss = -mean_log_prob_pos[valid_mask].mean()

    if not torch.isfinite(loss):
        return z.new_tensor(0.0)
    return loss

class Train:
    def __init__(
        self,
        data,
        datatype,
        device,
        random_seed=2024,
        dim_input=3000,
        dim_output=64,
        Arg=None,
        use_alignment=True,
        use_dual_stream=True,
        use_dynamic_graph=True,
    ):

        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.use_alignment = use_alignment
        self.use_dual_stream = use_dual_stream
        self.use_dynamic_graph = use_dynamic_graph
        self.use_spatial = True
        self.use_semantic = True
        
        self.use_contrastive = True
        self.contrastive_weight = 0.2
        self.cl_alpha = 1.0
        self.cl_beta = 0.5
        self.tau_inst = 0.07
        self.tau_sup = 0.1
        self.warmup_ratio = 0.3
        self.use_hard_negative = False
        self.hard_k_hard = 2
        self.hard_k_easy = 2
        self.ari_eval_interval = 5
        self.track_epoch_ari = False
        if Arg is not None:
            if getattr(Arg, 'use_alignment', None) is not None:
                self.use_alignment = Arg.use_alignment
            if getattr(Arg, 'use_dual_stream', None) is not None:
                self.use_dual_stream = Arg.use_dual_stream
            if getattr(Arg, 'use_spatial', None) is not None:
                self.use_spatial = Arg.use_spatial
            if getattr(Arg, 'use_semantic', None) is not None:
                self.use_semantic = Arg.use_semantic
            if getattr(Arg, 'use_contrastive', None) is not None:
                self.use_contrastive = Arg.use_contrastive
            if getattr(Arg, 'contrastive_weight', None) is not None:
                self.contrastive_weight = Arg.contrastive_weight
            if getattr(Arg, 'use_dynamic_graph', None) is not None:
                self.use_dynamic_graph = Arg.use_dynamic_graph
            if getattr(Arg, 'cl_alpha', None) is not None:
                self.cl_alpha = Arg.cl_alpha
            if getattr(Arg, 'cl_beta', None) is not None:
                self.cl_beta = Arg.cl_beta
            if getattr(Arg, 'tau_inst', None) is not None:
                self.tau_inst = Arg.tau_inst
            elif getattr(Arg, 'tau', None) is not None:
                self.tau_inst = Arg.tau
            if getattr(Arg, 'tau_sup', None) is not None:
                self.tau_sup = Arg.tau_sup
            elif getattr(Arg, 'tau', None) is not None:
                self.tau_sup = Arg.tau
            if getattr(Arg, 'warmup_ratio', None) is not None:
                self.warmup_ratio = Arg.warmup_ratio
            if getattr(Arg, 'hard_negative', None) is not None:
                self.use_hard_negative = Arg.hard_negative
            if getattr(Arg, 'hard_k_hard', None) is not None:
                self.hard_k_hard = Arg.hard_k_hard
            if getattr(Arg, 'hard_k_easy', None) is not None:
                self.hard_k_easy = Arg.hard_k_easy
            if getattr(Arg, 'ari_eval_interval', None) is not None:
                self.ari_eval_interval = Arg.ari_eval_interval
            if getattr(Arg, 'track_epoch_ari', None) is not None:
                self.track_epoch_ari = Arg.track_epoch_ari
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to_dense().to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to_dense().to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to_dense().to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to_dense().to(self.device)

        if self.use_dynamic_graph:
            self.paramed_adj_omics1 = Parametered_Graph(self.adj_feature_omics1, self.device).to(self.device)
            self.paramed_adj_omics2 = Parametered_Graph(self.adj_feature_omics2, self.device).to(self.device)
            self.adj_feature_omics1_copy = copy.deepcopy(self.adj_feature_omics1)
            self.adj_feature_omics2_copy = copy.deepcopy(self.adj_feature_omics2)
        else:
            self.paramed_adj_omics1 = None
            self.paramed_adj_omics2 = None
            self.adj_feature_omics1_copy = None
            self.adj_feature_omics2_copy = None

        self.EMA_coeffi = 0.9
        self.K = 5
        self.T = 4
        self.arg = Arg

        # 已删除聚类对比学习模块，使用监督对比学习替代
        # self.clustering = R5(self.datatype, self.arg)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        gt_labels = self.data.get('gt_labels', None)
        self.gt_labels = None
        if gt_labels is not None and len(gt_labels) == self.adata_omics1.n_obs:
            self.gt_labels = torch.as_tensor(gt_labels, dtype=torch.long, device=self.device)
        gt_labels_eval = self.data.get('gt_labels_eval', None)
        self.gt_labels_eval = None
        if gt_labels_eval is not None and len(gt_labels_eval) == self.adata_omics1.n_obs:
            self.gt_labels_eval = torch.as_tensor(gt_labels_eval, dtype=torch.long, device=self.device)
        elif self.gt_labels is not None:
            self.gt_labels_eval = self.gt_labels

        self.gt_train_mask = None
        self.gt_test_mask = None
        gt_train_mask = self.data.get('gt_train_mask', None)
        gt_test_mask = self.data.get('gt_test_mask', None)
        if gt_train_mask is not None and len(gt_train_mask) == self.adata_omics1.n_obs:
            self.gt_train_mask = torch.as_tensor(gt_train_mask, dtype=torch.bool, device=self.device)
        if gt_test_mask is not None and len(gt_test_mask) == self.adata_omics1.n_obs:
            self.gt_test_mask = torch.as_tensor(gt_test_mask, dtype=torch.bool, device=self.device)
        self.n_clusters = getattr(Arg, 'n_clusters', None) if Arg is not None else None
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        #输出维度
        self.dim_output1
        self.dim_output2
        
        if self.datatype == 'SPOTS':
           self.epochs = 200
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
           self.weight_decay = 5e-3
           self.learning_rate = 0.01
           
        elif self.datatype == 'Stereo-CITE-seq':
           self.epochs = 300
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
           self.weight_decay = 5e-2
           self.learning_rate = 0.01
           
        elif self.datatype == '10x':
           self.learning_rate = 0.01
           self.epochs = 30
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
           self.weight_decay = 5e-3
           self.EMA_coeffi = Arg.alpha
            
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
           self.epochs = 300
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
           self.learning_rate = 0.01
           self.weight_decay = 5e-2

        if Arg is not None:
            if getattr(Arg, 'epochs', None) is not None:
                self.epochs = Arg.epochs
            if getattr(Arg, 'learning_rate', None) is not None:
                self.learning_rate = Arg.learning_rate
    
    def train(self):
        """执行一次完整训练并返回结果字典。

        返回值（dict）包含：
        - 'emb_latent_omics1': numpy array, 第一模态归一化嵌入，shape (N, dim_output)
        - 'emb_latent_omics2': numpy array, 第二模态归一化嵌入，shape (N, dim_output)
        - 'SpaAlign': numpy array, 融合表示（emb_latent_combined），shape (N, dim_output)
        - 'adj_feature_omics1': numpy array, 学习后的特征邻接（用于记录/分析）

        注意：训练过程中会把重构损失、聚类损失、Frobenius 损失和（可选）扩散损失组合起来。
        """

        self.model = Encoder_overall(
            self.dim_input1, self.dim_output1, 
            self.dim_input2, self.dim_output2,
            use_alignment=self.use_alignment,
            use_dual_stream=self.use_dual_stream,
            use_spatial=self.use_spatial,
            use_semantic=self.use_semantic,
        ).to(self.device)
        if self.use_dynamic_graph:
            optim_params = (
                list(self.model.parameters())
                + list(self.paramed_adj_omics1.parameters())
                + list(self.paramed_adj_omics2.parameters())
            )
        else:
            optim_params = list(self.model.parameters())

        self.optimizer = torch.optim.SGD(
            optim_params,
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.model.train()
        warmup_epochs = max(1, int(self.epochs * self.warmup_ratio))
        loss_history = []
        recon_history = []
        graph_history = []
        inst_history = []
        sup_history = []
        total_cl_history = []
        beta_history = []
        ari_history = []
        ari_test_history = []
        best_epoch_ari = None
        best_epoch_ari_test = None
        best_epoch_idx = -1
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2,
            )

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            loss = self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2

            if self.use_dynamic_graph:
                updated_adj_omics1 = self.paramed_adj_omics1()
                updated_adj_omics2 = self.paramed_adj_omics2()

                loss_fro = (
                    torch.norm(updated_adj_omics1 - self.adj_feature_omics1_copy.detach(), p='fro')
                    + torch.norm(updated_adj_omics2 - self.adj_feature_omics2_copy.detach(), p='fro')
                ) / 2
            else:
                loss_fro = 0.0

            contrastive_loss_inst = self.features_omics1.new_tensor(0.0)
            contrastive_loss_sup = self.features_omics1.new_tensor(0.0)
            beta_t = 0.0
            if self.use_contrastive and 'aligned_rna' in results:
                contrastive_loss_inst = info_nce_inst(
                    results['aligned_rna'],
                    results['emb_latent_omics2'],
                    tau=self.tau_inst,
                )

            if self.use_contrastive and self.gt_labels is not None:
                if epoch < warmup_epochs:
                    beta_t = self.cl_beta * float(epoch + 1) / float(warmup_epochs)
                else:
                    beta_t = self.cl_beta
                contrastive_loss_sup = sup_infonce(
                    results['emb_latent_combined'],
                    self.gt_labels,
                    tau=self.tau_sup,
                    use_hard_negative=self.use_hard_negative,
                    k_hard=self.hard_k_hard,
                    k_easy=self.hard_k_easy,
                )

            contrastive_loss = self.cl_alpha * contrastive_loss_inst + beta_t * contrastive_loss_sup
            if not torch.isfinite(contrastive_loss):
                contrastive_loss = self.features_omics1.new_tensor(0.0)
            loss = loss + self.contrastive_weight * contrastive_loss

            loss = loss + loss_fro

            if not torch.isfinite(loss):
                loss = self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2 + loss_fro

            if self.track_epoch_ari and self.gt_labels_eval is not None and self.n_clusters is not None:
                if (epoch + 1) % max(1, self.ari_eval_interval) == 0 or (epoch + 1) == self.epochs:
                    with torch.no_grad():
                        emb_for_eval = F.normalize(results['emb_latent_combined'].detach(), p=2, eps=1e-12, dim=1).cpu().numpy()
                    pred = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', random_state=self.random_seed).fit_predict(emb_for_eval)
                    gt_eval_np = self.gt_labels_eval.detach().cpu().numpy()
                    labeled_mask = gt_eval_np >= 0
                    if labeled_mask.sum() > 1:
                        epoch_ari = adjusted_rand_score(gt_eval_np[labeled_mask], pred[labeled_mask])
                    else:
                        epoch_ari = np.nan

                    if self.gt_test_mask is not None:
                        test_mask_np = self.gt_test_mask.detach().cpu().numpy() & labeled_mask
                        if test_mask_np.sum() > 1:
                            epoch_ari_test = adjusted_rand_score(gt_eval_np[test_mask_np], pred[test_mask_np])
                        else:
                            epoch_ari_test = np.nan
                    else:
                        epoch_ari_test = np.nan
                else:
                    epoch_ari = np.nan
                    epoch_ari_test = np.nan
            else:
                epoch_ari = np.nan
                epoch_ari_test = np.nan

            loss_history.append(float(loss.detach().cpu().item()))
            recon_history.append(float((self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2).detach().cpu().item()))
            graph_history.append(float(loss_fro.detach().cpu().item()) if torch.is_tensor(loss_fro) else float(loss_fro))
            inst_history.append(float(contrastive_loss_inst.detach().cpu().item()))
            sup_history.append(float(contrastive_loss_sup.detach().cpu().item()))
            total_cl_history.append(float(contrastive_loss.detach().cpu().item()))
            beta_history.append(float(beta_t))
            ari_history.append(float(epoch_ari) if not np.isnan(epoch_ari) else np.nan)
            ari_test_history.append(float(epoch_ari_test) if not np.isnan(epoch_ari_test) else np.nan)

            if not np.isnan(epoch_ari):
                if best_epoch_ari is None or epoch_ari > best_epoch_ari:
                    best_epoch_ari = float(epoch_ari)
                    best_epoch_idx = epoch + 1
            if not np.isnan(epoch_ari_test):
                if best_epoch_ari_test is None or epoch_ari_test > best_epoch_ari_test:
                    best_epoch_ari_test = float(epoch_ari_test)

            print(loss, loss_fro, contrastive_loss)

            self.optimizer.zero_grad()
            loss.backward()
            # 全局梯度裁剪，防止扩散和对齐模块导致的梯度爆炸
            if self.use_dynamic_graph:
                clip_params = (
                    list(self.model.parameters())
                    + list(self.paramed_adj_omics1.parameters())
                    + list(self.paramed_adj_omics2.parameters())
                )
            else:
                clip_params = list(self.model.parameters())

            torch.nn.utils.clip_grad_norm_(
                clip_params,
                max_norm=5.0,
            )
            self.optimizer.step()
            # scheduler.step()


            if self.use_dynamic_graph:
                self.adj_feature_omics1 = self.paramed_adj_omics1()
                self.adj_feature_omics2 = self.paramed_adj_omics2()

                self.adj_feature_omics1_copy = self.EMA_coeffi * self.adj_feature_omics1_copy + (
                    1 - self.EMA_coeffi) * self.adj_feature_omics1.detach().clone()
                self.adj_feature_omics2_copy = self.EMA_coeffi * self.adj_feature_omics2_copy + (
                    1 - self.EMA_coeffi) * self.adj_feature_omics2.detach().clone()

        print("Model training finished!\n")

        start_time = time.time()
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

        end_time = time.time()
        print("Infer time: ", end_time - start_time)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'SpaAlign': emb_combined.detach().cpu().numpy(),
                  'adj_feature_omics1': self.adj_feature_omics1.detach().cpu().numpy()
        }
        output['training_log'] = {
            'loss_total': loss_history,
            'loss_recon': recon_history,
            'loss_graph': graph_history,
            'loss_cl_inst': inst_history,
            'loss_cl_sup': sup_history,
            'loss_cl_total': total_cl_history,
            'beta_t': beta_history,
            'epoch_ari': ari_history,
            'epoch_ari_test': ari_test_history,
            'best_epoch_ari': best_epoch_ari,
            'best_epoch_ari_test': best_epoch_ari_test,
            'best_epoch': best_epoch_idx,
        }
        
        return output
    
class Parametered_Graph(nn.Module):
    def __init__(self, adj, device):
        super(Parametered_Graph, self).__init__()
        self.adj = adj
        self.device = device

        n = self.adj.shape[0]
        self.paramed_adj_omics = nn.Parameter(torch.FloatTensor(n, n))
        self.paramed_adj_omics.data.copy_(self.adj)

    def forward(self, A=None):
        if A is None:
            adj = (self.paramed_adj_omics + self.paramed_adj_omics.t()) / 2
        else:
            adj = (A + A.t()) / 2

        adj = nn.ReLU(inplace=True)(adj)
        normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj.to(self.device)

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx



    
    
      

    
        
    
    
