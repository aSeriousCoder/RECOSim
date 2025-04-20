import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from Modules.scoring_model.common import (
    SelfAttention,
    AdditiveAttention,
    FeedForward,
    ndcg,
    mrr,
    maap,
)


ENCODING_DIM = 512
NUM_NEGATIVE_SAMPLES = 5


class UserFollowScoreModel(pl.LightningModule):
    def __init__(self, args, hidden_dim=None):
        super().__init__()
        self.args = args
        if hidden_dim is None:
            hidden_dim = ENCODING_DIM
        self.self_attention = SelfAttention(hidden_dim)
        self.additive_attention = AdditiveAttention(hidden_dim)
        self.feed_forward = FeedForward(4*hidden_dim, 1)

    def forward(self, user_history, user_embedding, friend_history, friend_embedding):
        """
        user_history: [B, 20, 512]
        user_embedding: [B, 512]
        friend_history: [B, 20, 512]
        friend_embedding: [B, 512]
        """
        base_user, _ = self.self_attention(user_history.float())
        base_user, _ = self.additive_attention(base_user)
        friend, _ = self.self_attention(friend_history.float())
        friend, _ = self.additive_attention(friend)
        return self.feed_forward(torch.cat([base_user, user_embedding, friend, friend_embedding], dim=1)).squeeze(-1)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        user_history, user_embedding, positive_friend_history, positive_friend_embedding, negative_friend_history, negative_friend_embedding = batch
        base_user, _ = self.self_attention(user_history.float())
        base_user, _ = self.additive_attention(base_user)
        positive_friend, _ = self.self_attention(positive_friend_history.float())
        positive_friend, _ = self.additive_attention(positive_friend)
        negative_friend, _ = self.self_attention(negative_friend_history.float().reshape(-1, user_history.shape[1], ENCODING_DIM))
        negative_friend, _ = self.additive_attention(negative_friend)
        negative_friend = negative_friend.reshape( -1, NUM_NEGATIVE_SAMPLES, ENCODING_DIM)
        pos_scores = self.feed_forward(torch.cat([base_user, user_embedding, positive_friend, positive_friend_embedding], dim=1))  # Pos Score
        neg_scores = self.feed_forward(
            torch.cat(
                [
                    base_user.unsqueeze(1).expand(-1, NUM_NEGATIVE_SAMPLES, -1).reshape(-1, ENCODING_DIM),
                    user_embedding.unsqueeze(1).expand(-1, NUM_NEGATIVE_SAMPLES, -1).reshape(-1, ENCODING_DIM),
                    negative_friend.reshape(-1, ENCODING_DIM),
                    negative_friend_embedding.reshape(-1, ENCODING_DIM),
                ],
                dim=1,
            )
        ).reshape(-1, NUM_NEGATIVE_SAMPLES)  # Neg Score
        pred = torch.cat([pos_scores.reshape(-1, 1), neg_scores.reshape(-1, NUM_NEGATIVE_SAMPLES)], dim=1)
        loss = (-torch.log_softmax(pred, dim=1).select(1, 0))
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        user_history, user_embedding, positive_friend_history, positive_friend_embedding, negative_friend_history, negative_friend_embedding = batch
        base_user, _ = self.self_attention(user_history.float())
        base_user, _ = self.additive_attention(base_user)
        positive_friend, _ = self.self_attention(positive_friend_history.float())
        positive_friend, _ = self.additive_attention(positive_friend)
        negative_friend, _ = self.self_attention(negative_friend_history.float().reshape(-1, user_history.shape[1], ENCODING_DIM))
        negative_friend, _ = self.additive_attention(negative_friend)
        negative_friend = negative_friend.reshape( -1, NUM_NEGATIVE_SAMPLES, ENCODING_DIM)
        pos_scores = self.feed_forward(torch.cat([base_user, user_embedding, positive_friend, positive_friend_embedding], dim=1))  # Pos Score
        neg_scores = self.feed_forward(
            torch.cat(
                [
                    base_user.unsqueeze(1).expand(-1, NUM_NEGATIVE_SAMPLES, -1).reshape(-1, ENCODING_DIM),
                    user_embedding.unsqueeze(1).expand(-1, NUM_NEGATIVE_SAMPLES, -1).reshape(-1, ENCODING_DIM),
                    negative_friend.reshape(-1, ENCODING_DIM),
                    negative_friend_embedding.reshape(-1, ENCODING_DIM),
                ],
                dim=1,
            )
        ).reshape(-1, NUM_NEGATIVE_SAMPLES)  # Neg Score
        pred = torch.cat([pos_scores.reshape(-1, 1), neg_scores.reshape(-1, NUM_NEGATIVE_SAMPLES)], dim=1)
        loss = (-torch.log_softmax(pred, dim=1).select(1, 0))
        self.log("val_loss", loss.mean())
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)
        ndcg_score = ndcg(all_scores, all_labels, k=all_scores.shape[1])
        self.log("val_ndcg", ndcg_score.mean())
        mrr_score = mrr(all_scores, all_labels)
        self.log("val_mrr", mrr_score.mean())
        map_score = maap(all_scores, all_labels)
        self.log("val_map", map_score.mean())

    def test_step(self, batch, batch_idx):
        user_history, user_embedding, positive_friend_history, positive_friend_embedding, negative_friend_history, negative_friend_embedding = batch
        base_user, _ = self.self_attention(user_history.float())
        base_user, _ = self.additive_attention(base_user)
        positive_friend, _ = self.self_attention(positive_friend_history.float())
        positive_friend, _ = self.additive_attention(positive_friend)
        negative_friend, _ = self.self_attention(negative_friend_history.float().reshape(-1, user_history.shape[1], ENCODING_DIM))
        negative_friend, _ = self.additive_attention(negative_friend)
        negative_friend = negative_friend.reshape( -1, NUM_NEGATIVE_SAMPLES, ENCODING_DIM)
        pos_scores = self.feed_forward(torch.cat([base_user, user_embedding, positive_friend, positive_friend_embedding], dim=1))  # Pos Score
        neg_scores = self.feed_forward(
            torch.cat(
                [
                    base_user.unsqueeze(1).expand(-1, NUM_NEGATIVE_SAMPLES, -1).reshape(-1, ENCODING_DIM),
                    user_embedding.unsqueeze(1).expand(-1, NUM_NEGATIVE_SAMPLES, -1).reshape(-1, ENCODING_DIM),
                    negative_friend.reshape(-1, ENCODING_DIM),
                    negative_friend_embedding.reshape(-1, ENCODING_DIM),
                ],
                dim=1,
            )
        ).reshape(-1, NUM_NEGATIVE_SAMPLES)  # Neg Score
        pred = torch.cat([pos_scores.reshape(-1, 1), neg_scores.reshape(-1, NUM_NEGATIVE_SAMPLES)], dim=1)
        loss = (-torch.log_softmax(pred, dim=1).select(1, 0))
        self.log("test_loss", loss.mean())
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)
        ndcg_score = ndcg(all_scores, all_labels, k=all_scores.shape[1])
        self.log("test_ndcg", ndcg_score.mean())
        mrr_score = mrr(all_scores, all_labels)
        self.log("val_mrr", mrr_score.mean())
        map_score = maap(all_scores, all_labels)
        self.log("val_map", map_score.mean())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

