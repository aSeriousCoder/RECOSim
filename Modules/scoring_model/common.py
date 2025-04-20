import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from sklearn.metrics import average_precision_score

class ParamlessSelfAttention():
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x):
        """
        x: 输入的特征矩阵，维度为(batch_size, seq_len, hidden_dim)
        """
        # 获取batch_size和seq_len
        batch_size, seq_len, hidden_dim = x.size()

        # 得到Query, Key, Value
        Q = x  # (batch_size, seq_len, hidden_dim)
        K = x  # (batch_size, seq_len, hidden_dim)
        V = x  # (batch_size, seq_len, hidden_dim)

        # 计算注意力分数（内积）
        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行缩放
        scaled_scores = scores / self.scale  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行softmax，得到注意力权重
        attn_weights = torch.softmax(
            scaled_scores, dim=-1
        )  # (batch_size, seq_len, seq_len)

        # 将注意力权重与Value相乘，得到self-attention后的表示
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        return attn_output, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim=512):
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim
        # Query, Key, Value参数矩阵
        self.query_matrix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim, bias=False), 
            # nn.Tanh(),
        )
        self.key_matrix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim, bias=False), 
            # nn.Tanh(),
        )
        self.value_matrix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim, bias=False), 
            # nn.Tanh(),
        )

        # Dropout层
        self.dropout = nn.Dropout(0.1)

        # 注意力分数归一化的比例系数
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x):
        """
        x: 输入的特征矩阵，维度为(batch_size, seq_len, hidden_dim)
        """
        # 获取batch_size和seq_len
        batch_size, seq_len, hidden_dim = x.size()

        # 得到Query, Key, Value
        Q = self.query_matrix(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key_matrix(x)  # (batch_size, seq_len, hidden_dim)
        V = self.value_matrix(x)  # (batch_size, seq_len, hidden_dim)

        # 计算注意力分数（内积）
        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行缩放
        scaled_scores = scores / self.scale  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行softmax，得到注意力权重
        attn_weights = torch.softmax(
            scaled_scores, dim=-1
        )  # (batch_size, seq_len, seq_len)

        # 对注意力权重进行dropout
        attn_weights = self.dropout(attn_weights)

        # 将注意力权重与Value相乘，得到self-attention后的表示
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        return attn_output, attn_weights


class AdditiveAttention(torch.nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            # nn.Tanh(),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, seq_dim]
        Returns:
            outputs, weights: [B, seq_dim], [B, seq_len]
        """
        weights = self.proj_v(self.proj(context)).squeeze(-1)  # [B, 1, seq_len]
        weights = torch.softmax(weights, dim=-1)  # [B, seq_len]
        return (
            torch.bmm(weights.unsqueeze(1), context).squeeze(1),
            weights,
        )  # [B, seq_dim], [B, seq_len]


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim=512, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            # nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, hidden_dim]
        Returns:
            outputs: [B, out_dim]
        """
        return self.proj(context)


class VariationalFeedForward(torch.nn.Module):
    def __init__(self, prior_mu, prior_logvar, hidden_dim=512, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, 2*out_dim)
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.prior_mu = prior_mu
        self.prior_logvar = prior_logvar

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, hidden_dim]
        Returns:
            outputs: [B, out_dim]
        """
        pj = self.proj(context)
        return reparametrize(pj[:, :self.out_dim], pj[:, self.out_dim:]), kl_divergence(pj[:, :self.out_dim], pj[:, self.out_dim:], self.prior_mu, self.prior_logvar)


def ndcg(scores, labels, k=6):
    scores = scores.cpu()
    labels = labels.cpu()
    # 降序排列，获取推荐列表的id
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    # 获取相关性得分，也就是0，1,如果命中
    hits = labels.gather(1, cut)
    # 计算位置关系，从2开始计
    position = torch.arange(2, 2+k)
    # 根据位置关系计算位置权重
    weights = 1 / torch.log2(position+1)
    # 计算DCG
    dcg = (hits * weights).sum(1)
    # 计算iDCG，由于相关性得分为0，1，且经过排序，所以计算前面为1对应weights之和即可。
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg

def mrr(scores, labels):
    labels = labels.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    mrr_scores = []
    for i in range(scores.shape[0]):
        sorted_indices = np.argsort(scores[i])[::-1]
        for j, idx in enumerate(sorted_indices):
            if labels[i][idx] == 1:
                mrr_scores.append(1 / (j + 1))
                break
    return torch.tensor(np.mean(mrr_scores))

def maap(scores, labels):
    labels = labels.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    map_scores = []
    for i in range(scores.shape[0]):
        map_score = average_precision_score(labels[i], scores[i])
        map_scores.append(map_score)
    return torch.tensor(np.mean(map_scores))

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def kl(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


def kl_divergence(mu_1, logvar_1, mu_2, logvar_2):
    """KL(P||Q)"""
    mu_2 = mu_2.to(mu_1.device)
    logvar_2 = logvar_2.to(logvar_1.device)
    kl = (logvar_2 - logvar_1) - 0.5
    kl += 0.5 * (torch.exp(2. * logvar_1) + ((mu_1 - mu_2)**2)) * torch.exp(-2. * logvar_2)
    return kl.sum(1).mean(0)
