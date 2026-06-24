"""SpaAlign 模型定义模块

包含多组学数据的编码器（`Encoder`/`Encoder_overall`）、解码器（`Decoder`）和融合 MLP（`MLP`）。
`Encoder_overall` 将两种组学分别编码、拼接潜在表示并通过 MLP 融合为最终的 `emb_latent_combined`。

实现细节说明（便于阅读与扩展）：
- `Encoder_overall` 中先将空间邻接与特征邻接扩展为带批次维度的张量（使用 `unsqueeze(0)`），按通道拼接后送入 `nn.Conv2d(in_channels=2, ...)` 做 1x1 卷积，随后 `squeeze(0)` 恢复原始形状；因此 Conv2d 的输入维度为 `(1, 2, N, N)`，输出为 `(1,1,N,N)`，再去掉 batch 维度得到 `(1,N,N)` 或 `(N,N)` 的邻接融合矩阵。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .dual_stream_alignment import DualStreamAlignment


class RNAQueryAligner(nn.Module):
    """
    RNAQueryAligner
    ----------------
    使用 RNA 模态的潜在表示作为 query，另一模态的潜在表示作为 key / value，
    做一次多头 cross-attention 并通过残差更新 RNA 表示，从而在细胞/spot 粒度上
    对不同组学的 latent 进行显式对齐。

    形状约定：
    - rna_latent:  (N, D)
    - other_latent:(N, D)
    返回：
    - aligned_rna: (N, D)，对齐后的 RNA latent
    - attn:        (N, N)，注意力权重矩阵（可用于解释）
    """

    def __init__(self, dim, n_heads=4, ff_dim=None, attn_dropout=0.0, alpha=1.0):
        super(RNAQueryAligner, self).__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.alpha = alpha

        # 线性投影得到 q / k / v
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # 输出投影
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.norm = nn.LayerNorm(dim)

        if ff_dim is None:
            ff_dim = dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, rna_latent, other_latent):
        # 输入 shape: (N, D) -> 视为 batch=1 的序列
        if rna_latent.dim() == 2:
            rna = rna_latent.unsqueeze(0)   # (1, N, D)
        else:
            rna = rna_latent
        if other_latent.dim() == 2:
            other = other_latent.unsqueeze(0)  # (1, N, D)
        else:
            other = other_latent

        b, n_rna, d = rna.shape
        _, n_other, _ = other.shape

        q = self.to_q(rna)   # (B, N_rna, D)
        k = self.to_k(other) # (B, N_other, D)
        v = self.to_v(other) # (B, N_other, D)

        # 拆分为多头
        q = q.view(b, n_rna, self.n_heads, self.head_dim).transpose(1, 2)      # (B, H, N_rna, Dh)
        k = k.view(b, n_other, self.n_heads, self.head_dim).transpose(1, 2)   # (B, H, N_other, Dh)
        v = v.view(b, n_other, self.n_heads, self.head_dim).transpose(1, 2)   # (B, H, N_other, Dh)

        # scaled dot-product attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale       # (B, H, N_rna, N_other)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                                           # (B, H, N_rna, Dh)
        out = out.transpose(1, 2).contiguous().view(b, n_rna, d)              # (B, N_rna, D)
        out = self.to_out(out)

        # 残差更新 + LayerNorm
        rna_updated = self.norm(rna + self.alpha * out)
        rna_updated = rna_updated + self.ffn_norm(self.ffn(rna_updated))

        # 去掉 batch 维
        rna_updated = rna_updated.squeeze(0)  # (N, D)

        # 为了可解释性，返回平均后的注意力矩阵 (N_rna, N_other)
        attn_mean = attn.mean(dim=1)  # (B, N_rna, N_other)
        attn_mean = attn_mean.squeeze(0)

        return rna_updated, attn_mean


class Encoder_overall(Module):
    """
    整体编码器类，用于处理两种组学数据的融合编码
    
    该类整合了空间邻接矩阵和特征邻接矩阵，通过编码器-解码器架构
    学习两种组学数据的潜在表示，并生成融合后的表示
    集成了扩散模型用于潜在表示的增强和去噪
    
    Parameters
    ----------
    dim_in_feat_omics1 : int
        第一种组学数据的输入特征维度
    dim_out_feat_omics1 : int
        第一种组学数据的输出特征维度
    dim_in_feat_omics2 : int
        第二种组学数据的输入特征维度
    dim_out_feat_omics2 : int
        第二种组学数据的输出特征维度
    dropout : float, optional
        Dropout概率，默认为0.0
    act : function, optional
        激活函数，默认为ReLU
    """
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2,
                 dropout=0.0, act=F.relu, use_alignment=True, use_dual_stream=True,
                 use_spatial=True, use_semantic=True):
        super(Encoder_overall, self).__init__()
        # 存储输入输出维度参数
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act
        # 是否使用 RNA-query 对齐模块（用于消融实验）
        self.use_alignment = use_alignment
        # 是否使用双流对齐
        self.use_dual_stream = use_dual_stream
        self.use_spatial = use_spatial
        self.use_semantic = use_semantic

        # 1x1卷积层：用于融合空间邻接矩阵和特征邻接矩阵（2通道->1通道）
        self.conv1X1_omics1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1X1_omics2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        # MLP层：用于融合两种组学数据的潜在表示
        # 输入维度为两种组学输出维度之和，输出维度与第一种组学输出维度相同
        # 原始简单拼接融合（保留为基线，可按需做 ablation）
        self.MLP = MLP(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1, self.dim_out_feat_omics1)

        # 双流对齐模块：以RNA为基准的空间和语义双流对齐
        if self.use_dual_stream:
            self.dual_stream_align = DualStreamAlignment(
                dim=self.dim_out_feat_omics1,
                heads=8,
                dim_head=32,
                dropout=dropout,
                alpha=0.5,
                use_spatial=self.use_spatial,
                use_semantic=self.use_semantic,
            )
        
        # RNA 作为 query、另一模态作为 key/value 的对齐模块（保留作为备选）
        # 使用 omics1 视作 RNA，omics2 视作另一组学
        if self.use_alignment and not self.use_dual_stream:
            self.aligner = RNAQueryAligner(
                dim=self.dim_out_feat_omics1,
                n_heads=4,
                ff_dim=self.dim_out_feat_omics1 * 2,
                attn_dropout=0.1,
                alpha=1.0,
            )
            # 对齐后再做一次小型 MLP 融合（可与原始 MLP 的结构保持一致）
            self.MLP_align = MLP(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        
        # 为每种组学数据创建编码器和解码器
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        
    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):
        """
        前向传播函数
        
        Parameters
        ----------
        features_omics1 : torch.Tensor
            第一种组学数据的特征矩阵
        features_omics2 : torch.Tensor
            第二种组学数据的特征矩阵
        adj_spatial_omics1 : torch.Tensor
            第一种组学数据的空间邻接矩阵
        adj_feature_omics1 : torch.Tensor
            第一种组学数据的特征邻接矩阵
        adj_spatial_omics2 : torch.Tensor
            第二种组学数据的空间邻接矩阵
        adj_feature_omics2 : torch.Tensor
            第二种组学数据的特征邻接矩阵
        
        Returns
        -------
        results : dict
            包含以下键的字典：
            - emb_latent_omics1: 第一种组学数据的潜在表示
            - emb_latent_omics2: 第二种组学数据的潜在表示
            - emb_latent_combined: 融合后的潜在表示
            - emb_recon_omics1: 第一种组学数据的重构表示
            - emb_recon_omics2: 第二种组学数据的重构表示
        """
        # 为邻接矩阵添加batch维度，shape: (1, N, N)
        _adj_spatial_omics1 = adj_spatial_omics1.unsqueeze(0)
        _adj_feature_omics1 = adj_feature_omics1.unsqueeze(0)

        _adj_spatial_omics2 = adj_spatial_omics2.unsqueeze(0)
        _adj_feature_omics2 = adj_feature_omics2.unsqueeze(0)

        # 沿通道维度拼接空间和特征邻接矩阵，shape: (2, N, N)
        cat_adj_omics1 = torch.cat((_adj_spatial_omics1, _adj_feature_omics1), dim=0)
        cat_adj_omics2 = torch.cat((_adj_spatial_omics2, _adj_feature_omics2), dim=0)

        # 使用1x1卷积融合两种邻接矩阵，然后移除batch维度
        adj_feature_omics1 = self.conv1X1_omics1(cat_adj_omics1).squeeze(0)
        adj_feature_omics2 = self.conv1X1_omics2(cat_adj_omics2).squeeze(0)

        # 对两种组学数据进行编码，得到特征嵌入和潜在表示
        feat_embeding1, emb_latent_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        feat_embeding2, emb_latent_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        # 原始简单拼接融合（作为基线，可用于 ablation）
        cat_emb_latent_base = torch.cat((emb_latent_omics1, emb_latent_omics2), dim=1)
        emb_latent_combined_base = self.MLP(cat_emb_latent_base)

        # ======================
        # 双流对齐：以RNA为基准的空间和语义对齐
        # ======================
        attn_rna_to_omics2 = None
        if self.use_dual_stream:
            # 使用双流对齐：空间对齐(基于邻接矩阵) + 语义对齐(基于特征相似度)
            aligned_rna = self.dual_stream_align(
                emb_latent_omics1,  # RNA作为基准
                emb_latent_omics2,  # 其他模态
                adj_spatial_omics1  # 空间邻接矩阵
            )
            # 简单拼接融合
            cat_emb_latent_align = torch.cat((aligned_rna, emb_latent_omics2), dim=1)
            emb_latent_combined = self.MLP(cat_emb_latent_align)
        elif self.use_alignment:
            # 使用原始RNAQueryAligner（向后兼容）
            aligned_rna, attn_rna_to_omics2 = self.aligner(emb_latent_omics1, emb_latent_omics2)
            cat_emb_latent_align = torch.cat((aligned_rna, emb_latent_omics2), dim=1)
            emb_latent_combined = self.MLP_align(cat_emb_latent_align)
        else:
            # 不使用对齐模块时，退化为原始简单融合
            emb_latent_combined = emb_latent_combined_base

        # 使用融合后的潜在表示和空间邻接矩阵进行解码重构
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # 组织返回结果
        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
        }
        # 为生物学解释性提供的注意力矩阵（训练时可忽略）
        if attn_rna_to_omics2 is not None:
            results['attn_rna_to_omics2'] = attn_rna_to_omics2

        # 为对比学习提供对齐后的特征
        if self.use_dual_stream:
            results['aligned_rna'] = aligned_rna if 'aligned_rna' in locals() else emb_latent_omics1

        return results     

'''
---------------------
Encoder & Decoder functions
author: Yahui Long https://github.com/JinmiaoChenLab/SpatialGlue
AGPL-3.0 LICENSE
---------------------
'''

class Encoder(Module): 
    
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Latent representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        # 可学习的权重矩阵，用于特征变换
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        # 初始化权重参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """使用Xavier均匀分布初始化权重矩阵"""
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        """
        前向传播
        
        Parameters
        ----------
        feat : torch.Tensor
            输入特征矩阵
        adj : torch.Tensor
            邻接矩阵（稀疏或密集）
        
        Returns
        -------
        feat_embeding : torch.Tensor
            特征嵌入（变换后的特征）
        x : torch.Tensor
            经过图卷积后的潜在表示
        """
        # 线性变换：特征矩阵与权重矩阵相乘
        feat_embeding = torch.mm(feat, self.weight)
        # 图卷积：邻接矩阵与特征嵌入相乘（稀疏矩阵乘法）
        x = torch.spmm(adj, feat_embeding)
        
        return feat_embeding, x
    
class Decoder(Module):
    
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Reconstructed representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        # 可学习的权重矩阵，用于特征变换
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        # 初始化权重参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """使用Xavier均匀分布初始化权重矩阵"""
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        """
        前向传播（解码过程）
        
        Parameters
        ----------
        feat : torch.Tensor
            输入特征矩阵（通常是潜在表示）
        adj : torch.Tensor
            邻接矩阵（稀疏或密集）
        
        Returns
        -------
        x : torch.Tensor
            重构后的特征表示
        """
        # 线性变换：特征矩阵与权重矩阵相乘
        x = torch.mm(feat, self.weight)
        # 图卷积：邻接矩阵与变换后的特征相乘（稀疏矩阵乘法）
        x = torch.spmm(adj, x)
        
        return x                  


class MLP(nn.Module):
    """
    多层感知机（MLP）网络
    
    用于融合不同组学数据的潜在表示。当前实现为两层全连接网络，
    激活函数和dropout层已被注释掉。
    
    Parameters
    ----------
    input_size : int
        输入特征维度
    hidden_size : int
        隐藏层特征维度
    output_size : int
        输出特征维度
    dropout_rate : float, optional
        Dropout概率，默认为0.5（当前未使用）
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        # 第一层全连接层：输入层到隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 激活函数和dropout层已被注释（可根据需要启用）
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout_rate)
        # 第二层全连接层：隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入特征矩阵
        
        Returns
        -------
        out : torch.Tensor
            输出特征矩阵
        """
        # 第一层全连接变换
        out = self.fc1(x)
        # 激活函数和dropout（当前未使用）
        # out = self.relu(out)
        # out = self.dropout(out)
        # 第二层全连接变换
        out = self.fc2(out)
        return out
