import torch
import numpy as np
import pandas as pd
import dgl
from tqdm import tqdm
from Simulation.agent.recsys.base_recsys import BaseRecSys


class PopRecSys(BaseRecSys):
    def __init__(self, config, logger, graph):
        self._config = config
        self._logger = logger
        self._graph = graph
        self._post_rec_list_length = config.post_rec_list_length
        self._friend_rec_list_length = config.friend_rec_list_length
        self.msg_sender = MsgSender()

        self.msg_sender.mailbox['begin_day'] = self._config.padding
        self.msg_sender.mailbox['end_day'] = self._config.padding + self._config.window_size
        id_edge_like_up_to_now = self._graph.graph.filter_edges(self.msg_sender.interactions_in_window, etype='like').int()
        id_edge_comment_up_to_now = self._graph.graph.filter_edges(self.msg_sender.interactions_in_window, etype='comment_on').int()
        id_edge_repost_up_to_now = self._graph.graph.filter_edges(self.msg_sender.interactions_in_window, etype='repost_on').int()
        self.interaction_sub_g = dgl.edge_subgraph(self._graph.graph, {
            ('user', 'like', 'post'): id_edge_like_up_to_now,
            ('user', 'comment_on', 'post'): id_edge_comment_up_to_now,
            ('user', 'repost_on', 'post'): id_edge_repost_up_to_now
        }, relabel_nodes=False, store_ids=True)

        self.msg_sender.mailbox['begin_day'] = 0
        self.msg_sender.mailbox['end_day'] = 1
        id_edge_follow_up_to_now = self._graph.graph.filter_edges(self.msg_sender.interactions_in_window, etype='follow').int()
        self.follow_sub_g = dgl.edge_subgraph(self._graph.graph, {
            ('user', 'follow', 'user'): id_edge_follow_up_to_now,
        }, relabel_nodes=False, store_ids=True)

    def prepare(self):
        pass

    def recommend_post(self):
        self._logger.debug("Recommending Popular News ...")
        # window中采样
        self.msg_sender.mailbox['begin_day'] = self._graph._cur_tag - self._config.window_size
        self.msg_sender.mailbox['end_day'] = self._graph._cur_tag
        id_post_up_to_now = self._graph.graph.filter_edges(self.msg_sender.interactions_in_window, etype='create_post').int()
        popularity = (
            self._graph.graph.in_degrees(etype="like")[id_post_up_to_now.long()]
            + self._graph.graph.in_degrees(etype="repost_of")[id_post_up_to_now.long()]
            + self._graph.graph.in_degrees(etype="comment_of")[id_post_up_to_now.long()]
        ).float()
        top_pop = torch.topk(popularity, self._post_rec_list_length * 10)
        top_pop_indices = id_post_up_to_now[top_pop.indices]
        top_pop_values = top_pop.values
        candidate_mat = torch.stack([top_pop_indices] * self._graph.graph.num_nodes('user'), dim=0)

        candidate_mat_in_pair = torch.stack([
            torch.tensor(list(range(self._graph.graph.num_nodes('user')))).unsqueeze(1).expand(-1, self._post_rec_list_length * 10).reshape(-1), 
            candidate_mat.reshape(-1)
        ], dim=1)
        filter_out_mat_1 = self._graph.graph.has_edges_between(candidate_mat_in_pair[:,0].int(), candidate_mat_in_pair[:,1].int(), ('user', 'post_rec', 'post')).reshape(candidate_mat.shape)
        filter_out_mat_2 = self._graph.graph.has_edges_between(candidate_mat_in_pair[:,0].int(), candidate_mat_in_pair[:,1].int(), ('user', 'create_post', 'post')).reshape(candidate_mat.shape)
        in_init_musk = candidate_mat_in_pair[:, 1] < self.interaction_sub_g.num_nodes('post')
        filter_out_mat_3 = torch.zeros(candidate_mat.reshape(-1).shape).bool()
        filter_out_mat_3[in_init_musk] = self.interaction_sub_g.has_edges_between(candidate_mat_in_pair[in_init_musk][:,0].int(), candidate_mat_in_pair[in_init_musk][:,1].int(), ('user', 'like', 'post'))
        filter_out_mat_3 = filter_out_mat_3.reshape(candidate_mat.shape)
        filter_out_mat_4 = torch.zeros(candidate_mat.reshape(-1).shape).bool()
        filter_out_mat_4[in_init_musk] = self.interaction_sub_g.has_edges_between(candidate_mat_in_pair[in_init_musk][:,0].int(), candidate_mat_in_pair[in_init_musk][:,1].int(), ('user', 'comment_on', 'post'))
        filter_out_mat_4 = filter_out_mat_4.reshape(candidate_mat.shape)
        filter_out_mat_5 = torch.zeros(candidate_mat.reshape(-1).shape).bool()
        filter_out_mat_5[in_init_musk] = self.interaction_sub_g.has_edges_between(candidate_mat_in_pair[in_init_musk][:,0].int(), candidate_mat_in_pair[in_init_musk][:,1].int(), ('user', 'repost_on', 'post'))
        filter_out_mat_5 = filter_out_mat_5.reshape(candidate_mat.shape) 
        filter_out_mat = filter_out_mat_1 | filter_out_mat_2 | filter_out_mat_3 | filter_out_mat_4 | filter_out_mat_5

        popularity_ranking_score = torch.stack([top_pop_values] * self._graph.graph.num_nodes('user'), dim=0)
        selected = torch.topk((1-filter_out_mat.float())*popularity_ranking_score, dim=1, k=self._post_rec_list_length).indices
        pop_rec_list = candidate_mat.gather(1, selected)
        pop_rec_list_in_pair = torch.stack([
            torch.tensor(list(range(self._graph.graph.num_nodes('user')))).unsqueeze(1).expand(-1, self._post_rec_list_length).reshape(-1), 
            pop_rec_list.reshape(-1)
        ], dim=1)
        return pop_rec_list_in_pair

    def recommend_friends(self):
        self._logger.debug("Recommending Popular Friends ...")
        popularity = self._graph.graph.in_degrees(etype="follow").float()
        top_pop = torch.topk(popularity, self._friend_rec_list_length * 10)
        top_pop_indices = top_pop.indices
        top_pop_values = top_pop.values
        candidate_mat = torch.stack([top_pop_indices] * self._graph.graph.num_nodes('user'), dim=0)

        candidate_mat_in_pair = torch.stack([
            torch.tensor(list(range(self._graph.graph.num_nodes('user')))).unsqueeze(1).expand(-1, self._friend_rec_list_length * 10).reshape(-1), 
            candidate_mat.reshape(-1)
        ], dim=1)
        filter_out_mat_1 = self._graph.graph.has_edges_between(candidate_mat_in_pair[:,0].int(), candidate_mat_in_pair[:,1].int(), ('user', 'friend_rec', 'user')).reshape(candidate_mat.shape)
        filter_out_mat_2 = self.follow_sub_g.has_edges_between(candidate_mat_in_pair[:,0].int(), candidate_mat_in_pair[:,1].int(), ('user', 'follow', 'user')).reshape(candidate_mat.shape)
        filter_out_mat = filter_out_mat_1 | filter_out_mat_2

        popularity_ranking_score = torch.stack([top_pop_values] * self._graph.graph.num_nodes('user'), dim=0)
        selected = torch.topk((1-filter_out_mat.float())*popularity_ranking_score, dim=1, k=self._friend_rec_list_length).indices
        pop_rec_list = candidate_mat.gather(1, selected)
        pop_rec_list_in_pair = torch.stack([
            torch.tensor(list(range(self._graph.graph.num_nodes('user')))).unsqueeze(1).expand(-1, self._friend_rec_list_length).reshape(-1), 
            pop_rec_list.reshape(-1)
        ], dim=1)
        return pop_rec_list_in_pair


class MsgSender:
    def __init__(self):
        self.mailbox = {}

    def interactions_in_window(self, edges):
        begin_day = self.mailbox["begin_day"]
        end_day = self.mailbox["end_day"]
        edge_tag = edges.data['tag']
        return (edge_tag >= begin_day) & (edge_tag < end_day)
