import torch
import numpy as np
import pandas as pd
import dgl
from tqdm import tqdm
from Simulation.data.graph import SimuLineGraph
from Simulation.agent.recsys.base_recsys import BaseRecSys
from Simulation.agent.recsys.deep_recsys import DeepRecSys
from Simulation.agent.recsys.random_recsys import RandomRecSys
from Simulation.agent.recsys.pop_recsys import PopRecSys
from Simulation.agent.recsys.social_recsys import SocialRecSys


class MixRecSys(BaseRecSys):
    '''
    Post Rec: Rand (avg interaction count == 0) -> Deep (avg interaction count == 10)
    Friend Rec: Pop (friends count == 0) -> Social (friends count == 10)
    '''
    def __init__(self, config, logger, graph):
        self._config = config
        self._logger = logger
        self._graph = graph
        self._post_rec_list_length = config.post_rec_list_length
        self._friend_rec_list_length = config.friend_rec_list_length
        self._random_recsys = RandomRecSys(config, logger, graph)
        self._pop_recsys = PopRecSys(config, logger, graph)
        self._social_recsys = SocialRecSys(config, logger, graph)
        self._deep_recsys = DeepRecSys(config, logger, graph)
        self._call_recsys = {
            'Random': self._random_recsys,
            'Pop': self._pop_recsys,
            'Social': self._social_recsys,
            'Deep': self._deep_recsys
        }

    def recommend_post(self):
        rand_rec = self._random_recsys.recommend_post()
        deep_rec = self._deep_recsys.recommend_post()
        repost_counts = self._graph.graph.nodes["user"].data["repost_count"]
        comment_counts = self._graph.graph.nodes["user"].data["comment_count"]
        like_counts = self._graph.graph.nodes["user"].data["like_count"]
        interaction_counts = (repost_counts + comment_counts + like_counts).sum(dim=1)
        user_avg_interaction_count = interaction_counts.clip(min=0, max=10)
        num_from_deep = (user_avg_interaction_count / 10 * self._post_rec_list_length).int()

        num_users = self._graph.graph.num_nodes("user")
        rand_rec_dict = group_by_user(rand_rec)
        deep_rec_dict = group_by_user(deep_rec)
        final_rec_dict = {}
        for i in range(num_users):
            rec = []
            if i in deep_rec_dict:
                rec.extend(deep_rec_dict[i][:min(num_from_deep[i], len(deep_rec_dict[i]))])
            rec.extend(rand_rec_dict[i][: self._post_rec_list_length - len(rec)])
            final_rec_dict[i] = rec
        final_rec = []
        for i in range(num_users):
            final_rec.extend([[i, post_id] for post_id in final_rec_dict[i]])
        final_rec = torch.tensor(final_rec).long()
        return final_rec

    def recommend_friends(self):
        pop_rec = self._pop_recsys.recommend_friends()
        social_rec = self._social_recsys.recommend_friends()
        user_avg_friend_count = self._graph.graph.out_degrees(etype="follow").clip(min=0, max=10)
        num_from_social = (user_avg_friend_count / 10 * self._friend_rec_list_length).int()

        num_users = self._graph.graph.num_nodes("user")
        pop_rec_dict = group_by_user(pop_rec)
        social_rec_dict = group_by_user(social_rec)
        final_rec_dict = {}
        for i in range(num_users):
            rec = []
            if i in social_rec_dict:
                rec.extend(social_rec_dict[i][:min(num_from_social[i], len(social_rec_dict[i]))])
            rec.extend(pop_rec_dict[i][: self._friend_rec_list_length - len(rec)])
            final_rec_dict[i] = rec
        final_rec = []
        for i in range(num_users):
            final_rec.extend([[i, user_id] for user_id in final_rec_dict[i]])
        final_rec = torch.tensor(final_rec).long()
        return final_rec

    def prepare(self):
        self._random_recsys.prepare()
        self._pop_recsys.prepare()
        self._social_recsys.prepare()
        self._deep_recsys.prepare()


def group_by_user(rec):
    # rec: [user_id, post_id] * N
    # return: {user_id: [post_id] * M}
    rec_list = rec.tolist()
    rec_dict = {}
    for user_id, post_id in rec_list:
        if user_id not in rec_dict:
            rec_dict[user_id] = []
        rec_dict[user_id].append(post_id)
    return rec_dict
