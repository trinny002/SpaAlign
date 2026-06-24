"""
双流对齐模块 (Dual-Stream Alignment)
实现以RNA为基准的空间对齐和语义对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Compatibility wrapper for einsum/rearrange to support both einops and torch
try:
    from einops import rearrange as _rearrange, einsum as _einops_einsum

    def rearrange(tensor, pattern, **kwargs):
        return _rearrange(tensor, pattern, **kwargs)

    def einsum(*args):
        """
        Compatibility wrapper that accepts (tensors..., pattern)
        and calls einops.einsum(tensors..., pattern).
        """
        if len(args) < 2:
            raise TypeError('einsum requires at least two arguments')
        # extract pattern and tensors regardless of order
        if isinstance(args[-1], str):
            tensors = args[:-1]
            pattern = args[-1]
        elif isinstance(args[0], str):
            pattern = args[0]
            tensors = args[1:]
        else:
            # no clear pattern string; fallback to einops signature
            return _einops_einsum(*args)

        # if pattern looks like einops style (contains spaces), use einops
        if isinstance(pattern, str) and (' ' in pattern):
            return _einops_einsum(*tensors, pattern)
        # otherwise prefer torch.einsum compact pattern
        return torch.einsum(pattern, *tensors)
except ImportError:
    def rearrange(tensor, pattern, **kwargs):
        # 简单实现,仅用于兼容
        return tensor

    def einsum(*args):
        """
        Compatibility wrapper that accepts either (pattern, tensors...) as
        torch.einsum expects, or (tensors..., pattern) and will reorder.
        """
        if len(args) == 0:
            raise TypeError('einsum requires arguments')
        # If first arg is a string, assume torch.einsum signature
        if isinstance(args[0], str):
            pattern = args[0]
            tensors = args[1:]
            return torch.einsum(pattern, *tensors)
        # If last arg is a string, reorder
        if isinstance(args[-1], str):
            tensors = args[:-1]
            pattern = args[-1]
            return torch.einsum(pattern, *tensors)
        # otherwise, try to call torch.einsum directly (may raise)
        return torch.einsum(*args)


class SpatialAlignment(nn.Module):
    """
    空间对齐模块：基于空间邻接矩阵进行对齐
    以RNA为query,其他模态为key/value,利用空间邻接信息进行对齐
    """
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (dim == inner_dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        
        # 投影层
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, rna_latent, other_latent, spatial_adj):
        """
        Args:
            rna_latent: (N, D) RNA潜在表示
            other_latent: (N, D) 其他模态潜在表示
            spatial_adj: (N, N) 空间邻接矩阵
        Returns:
            aligned_rna: (N, D) 对齐后的RNA表示
        """
        # 添加batch维度: (1, N, D)
        if rna_latent.dim() == 2:
            rna = rna_latent.unsqueeze(0)
            other = other_latent.unsqueeze(0)
            spatial_adj = spatial_adj.unsqueeze(0)
        else:
            rna = rna_latent
            other = other_latent
            
        b, n, _, h = *rna.shape, self.heads
        
        # 投影
        q = self.to_q(rna)
        k = self.to_k(other)
        v = self.to_v(other)
        
        # 多头切分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        # 计算注意力
        # einops.einsum in some versions expects the pattern as the last argument,
        # so place tensors first and the pattern string last for compatibility.
        dots = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        # 应用空间邻接矩阵作为mask/权重
        spatial_mask = spatial_adj.unsqueeze(1)  # (B, 1, N, N)
        dots = dots + spatial_mask * 0.1  # 增强空间邻近节点的注意力

        attn = self.attend(dots)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 残差连接和归一化
        aligned_rna = self.norm(rna + self.to_out(out))
        
        # 移除batch维度
        if rna_latent.dim() == 2:
            aligned_rna = aligned_rna.squeeze(0)
            
        return aligned_rna


class SemanticAlignment(nn.Module):
    """
    语义对齐模块：基于特征相似度进行对齐
    以RNA为query,其他模态为key/value,利用特征相似度进行对齐
    """
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (dim == inner_dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        
        # 投影层
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, rna_latent, other_latent):
        """
        Args:
            rna_latent: (N, D) RNA潜在表示
            other_latent: (N, D) 其他模态潜在表示
        Returns:
            aligned_rna: (N, D) 对齐后的RNA表示
        """
        # 添加batch维度
        if rna_latent.dim() == 2:
            rna = rna_latent.unsqueeze(0)
            other = other_latent.unsqueeze(0)
        else:
            rna = rna_latent
            other = other_latent
            
        b, n, _, h = *rna.shape, self.heads
        
        # 投影
        q = self.to_q(rna)
        k = self.to_k(other)
        v = self.to_v(other)
        
        # 多头切分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # 计算注意力(基于语义相似度)
        # 与 SpatialAlignment 一致，确保 einops.einsum 的参数顺序兼容不同版本
        dots = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        attn = self.attend(dots)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 残差连接和归一化
        aligned_rna = self.norm(rna + self.to_out(out))
        
        # 移除batch维度
        if rna_latent.dim() == 2:
            aligned_rna = aligned_rna.squeeze(0)
            
        return aligned_rna


class DualStreamAlignment(nn.Module):
    """
    双流对齐模块：结合空间对齐和语义对齐
    以RNA为基准,对空间信息和语义信息进行双流对齐
    """
    def __init__(self, dim, heads=8, dim_head=32, dropout=0., alpha=0.5, use_spatial=True, use_semantic=True):
        super().__init__()
        self.alpha = alpha  # 空间和语义对齐的融合权重
        self.use_spatial = use_spatial
        self.use_semantic = use_semantic

        self.spatial_align = SpatialAlignment(dim, heads, dim_head, dropout) if self.use_spatial else None
        self.semantic_align = SemanticAlignment(dim, heads, dim_head, dropout) if self.use_semantic else None
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, rna_latent, other_latent, spatial_adj):
        """
        Args:
            rna_latent: (N, D) RNA潜在表示
            other_latent: (N, D) 其他模态潜在表示
            spatial_adj: (N, N) 空间邻接矩阵
        Returns:
            aligned_rna: (N, D) 双流对齐后的RNA表示
            spatial_attn: 空间对齐的注意力(可选)
        """
        # 允许按流做消融：only-spatial / only-semantic / dual-stream
        if self.use_spatial and self.use_semantic:
            spatial_aligned = self.spatial_align(rna_latent, other_latent, spatial_adj)
            semantic_aligned = self.semantic_align(rna_latent, other_latent)
            combined = torch.cat([spatial_aligned, semantic_aligned], dim=-1)
            aligned_rna = self.fusion(combined)
        elif self.use_spatial:
            aligned_rna = self.spatial_align(rna_latent, other_latent, spatial_adj)
        elif self.use_semantic:
            aligned_rna = self.semantic_align(rna_latent, other_latent)
        else:
            aligned_rna = rna_latent

        aligned_rna = self.norm(aligned_rna + rna_latent)  # 残差连接
        
        return aligned_rna

