"""
监督对比学习损失函数
适配多组学数据的对比学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比学习损失
    基于标签信息构建正负样本对
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: (N, D) 特征表示
            labels: (N,) 标签
        Returns:
            loss: 对比学习损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)  # (N, N)
        
        # 构建标签mask: 相同标签为正样本对
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (N, N)
        
        # 排除自身
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算exp相似度
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算每个样本的平均log概率(只考虑正样本)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # 损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        
        return loss


class MultiModalContrastiveLoss(nn.Module):
    """
    多模态对比学习损失
    用于对齐不同模态的表示
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, rna_features, other_features, labels=None):
        """
        Args:
            rna_features: (N, D) RNA特征
            other_features: (N, D) 其他模态特征
            labels: (N,) 可选标签,用于监督对比学习
        Returns:
            loss: 对比学习损失
        """
        device = rna_features.device
        batch_size = rna_features.shape[0]
        
        # 归一化
        rna_features = F.normalize(rna_features, p=2, dim=1)
        other_features = F.normalize(other_features, p=2, dim=1)
        
        # 计算跨模态相似度
        similarity_matrix = torch.matmul(rna_features, other_features.T)  # (N, N)
        
        if labels is not None:
            # 监督对比学习: 相同标签为正样本
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # 无监督: 对角线为正样本(同一细胞的不同模态)
            mask = torch.eye(batch_size).to(device)
        
        # 计算损失
        exp_logits = torch.exp(similarity_matrix / self.temperature)
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 只考虑正样本
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    用于多模态对比学习,支持自定义正负样本对
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels, indices_tuple=None):
        """
        Args:
            features: (N, D) 特征
            labels: (N,) 标签
            indices_tuple: (t1, p, t2, n) 正负样本索引元组
        Returns:
            loss: 对比学习损失
        """
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        
        if indices_tuple is not None:
            # 使用自定义的正负样本对
            t1, p, t2, n = indices_tuple
            anchors = features[t1]
            positives = features[p]
            negatives = features[n]
            
            # 计算正样本相似度
            pos_sim = torch.sum(anchors * positives, dim=1) / self.temperature
            
            # 计算负样本相似度
            neg_sim = torch.sum(anchors.unsqueeze(1) * negatives.unsqueeze(0), dim=-1) / self.temperature
            
            # 计算损失
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels_contrastive = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels_contrastive)
        else:
            # 标准对比学习
            similarity_matrix = torch.matmul(features, features.T) / self.temperature
            labels_matrix = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
            labels_matrix = labels_matrix - torch.eye(labels_matrix.shape[0], device=device)
            
            # 排除自身
            mask = torch.eye(similarity_matrix.shape[0], device=device)
            similarity_matrix = similarity_matrix - mask * 1e9
            
            # 计算损失
            exp_logits = torch.exp(similarity_matrix)
            log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (labels_matrix * log_prob).sum(1) / (labels_matrix.sum(1) + 1e-8)
            loss = -mean_log_prob_pos.mean()
            
        return loss

