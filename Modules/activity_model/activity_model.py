import pickle
import torch
import numpy as np


class UserActivityModel():
    def __init__(self, action_type):
        self.action_type = action_type
        
        if action_type == 'post':
            self.index = 0
        elif action_type == 'repost':
            self.index = 1
        elif action_type == 'comment':
            self.index = 2
        elif action_type == 'like':
            self.index = 3
        else:
            raise Exception('Action Type Not Implement !')
        
        with open(f'Modules/ckpts/activity_model_{action_type}.pkl', 'rb') as file:
            self.model = pickle.load(file)
    
    def predict(self, ugt_state, last_ugt_state, mutual_info_value):
        """
        @INPUT: feat - the round-difference of user ugt_state
        @OUTPUT: the predicted action density
        """
        action_density_this_round = ugt_state[:, 7 * self.index - 1]
        ugt_state_diff = ugt_state - last_ugt_state
        action_density_diff = self.model.predict(ugt_state_diff)
        action_density_next_round_by_model = action_density_this_round + action_density_diff
        action_density_next_round_by_poisson = torch.from_numpy(np.random.poisson(action_density_this_round.numpy()))
        action_density_next_round = action_density_next_round_by_model * mutual_info_value + action_density_next_round_by_poisson * (1 - mutual_info_value)
        return action_density_next_round
