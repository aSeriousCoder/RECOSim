import torch
import numpy as np
import pickle
from Modules.activity_model.activity_model import UserActivityModel
from Modules.generate_model.topic_gmm import TopicGMM


CONCENTRATE = 0.5


class UserActionService():
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        # Activity Models
        self.activity_models = {action_type: UserActivityModel(action_type) for action_type in ['post', 'repost', 'comment', 'like']}
        # Scoring Models
        self.scoring_models = {}
        self.quantile_values = {}
        for action_type in ['repost', 'comment', 'like', 'follow']:
            with open(f'Modules/ckpts/scoring_model_{action_type}.pkl', 'rb') as file:
                self.scoring_models[action_type] = pickle.load(file)
                self.scoring_models[action_type] = self.scoring_models[action_type].to(config.device)
                self.scoring_models[action_type].eval()
            with open(f'Modules/ckpts/quantile_values_{action_type}.pkl', 'rb') as file:
                self.quantile_values[action_type] = pickle.load(file)
        # Generate Models
        self.topic_gmm = TopicGMM()
        self.length_rec_list = config.post_rec_list_length
    
    def pred_action_density(self, action_type, ugt_state, last_ugt_state, mutual_info_value):
        density = self.activity_models[action_type].predict(ugt_state, last_ugt_state, mutual_info_value)
        density[density < 0] = 0
        density[density > self.length_rec_list] = self.length_rec_list
        return density.round().int()

    def score(self, action_type, batch):
        """
        For repost / comment / like
        user_history: [B, 20, 512]
        user_embedding: [B, 512]
        post_embedding: [B, 512]
        return: [B]
        ---
        For follow
        user_history: [B, 20, 512]
        user_embedding: [B, 512]
        friend_history: [B, 20, 512]
        friend_embedding: [B, 512]
        """
        with torch.no_grad():
            scores = self.scoring_models[action_type](*batch)
        return scores
    
    def generate(self, topic_ids):
        return self.topic_gmm.generate(topic_ids)

    def density_2_threshold(self, action_type, density):
        quantile_value = self.quantile_values[action_type]
        quantile_index = (100 - (100 * density / self.length_rec_list)).int()
        return quantile_value[quantile_index]

    def pred_post_topic_ID(self, topic_engage, src_topic_id=None):
        """
        @param topic_engage: [B, N], N is the number of topics
        @return topic_ID: [B]
        Sampling from the topic_engage distribution
        """
        return torch.multinomial(topic_engage.softmax(dim=1), 1).squeeze()

