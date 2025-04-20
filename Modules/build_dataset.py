import dgl
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import pandas as pd
from Simulation.agent.user.awesome_user import AwesomeUser
from Simulation.config import get_config
from Simulation.util.log import init_logger
from Simulation.data.graph import SimuLineGraph


def main():
    # === Configurations ===
    config = get_config()
    config.build_dataset = True
    # === Build Logger ===
    logger = init_logger(config)
    logger.info('Configurations & Logger Build!')
    # === Build Graph ===
    graph = SimuLineGraph(config, logger, force_reload=False)
    logger.info('Graph Build!')
    # === Build User ===
    builder = DatasetBuilder(config, logger, graph)
    logger.info('DatasetBuilder Build!')
    builder.collect()


DATASET_SAVING_DIR = "Modules/data/"


class DatasetBuilder(AwesomeUser):
    def __init__(self, config, logger, graph, user_action_service=None, metrics_service=None):
        super(DatasetBuilder, self).__init__(config, logger, graph, user_action_service, metrics_service)

    def collect(self):

        # Follow (which is not along with "tag")
        print('Building follow.pt')
        src_uid, dst_uid = self._graph.graph.edges(etype='follow')
        src_user_history_interacted_post_embeddings = self._graph.graph.nodes['user'].data['history_interacted'][src_uid]
        src_user_embeddings = self._graph.graph.nodes['user'].data['embedding'][src_uid]
        dst_user_history_interacted_post_embeddings = self._graph.graph.nodes['user'].data['history_interacted'][dst_uid]
        dst_user_embeddings = self._graph.graph.nodes['user'].data['embedding'][dst_uid]
        data = torch.cat([src_user_history_interacted_post_embeddings, src_user_embeddings.unsqueeze(1), dst_user_history_interacted_post_embeddings, dst_user_embeddings.unsqueeze(1)], dim=1)
        torch.save(data, DATASET_SAVING_DIR + "follow.pt")

        for i in range(7):  # [1007,1014)
            # Activity Model Dataset
            # Features: UGT State
            # Labels: Action Density This Round
            print(f"CUR TAG: {self._graph._cur_tag}")
            print(f'Building acticity_density_{self._graph._cur_tag}.pt')
            self.collect_action_density_this_round()
            ugt_state = self._graph.graph.nodes["user"].data["ugt_state"]
            action_density_this_round = torch.stack([
                self._graph.graph.nodes["user"].data["{}_count_this_round".format(action_name)] for action_name in ["post", "repost", "comment", "like"]
            ], dim=1)
            data = torch.cat([ugt_state, action_density_this_round], dim=1)
            torch.save(data, DATASET_SAVING_DIR + "acticity_density_{}.pt".format(self._graph._cur_tag))
            print(data.shape)
            # Scoring Model Dataset
            # Type: Repost Comment Like
            # Features: User History Interacted Post Embeddings, User Embeddings, Post Embeddings
            # Repost
            print(f'Building repost_{self._graph._cur_tag}.pt')
            uid, pid = self._graph.graph.edges(etype='repost_on')
            musk = self._graph.graph.edges['repost_on'].data['tag'] == self._graph._cur_tag
            uid = uid[musk]
            pid = pid[musk]
            user_history_interacted_post_embeddings = self._graph.graph.nodes['user'].data['history_interacted'][uid]
            user_embeddings = self._graph.graph.nodes['user'].data['embedding'][uid]
            post_embeddings = self._graph.graph.nodes['post'].data['embedding'][pid]
            data = torch.cat([user_history_interacted_post_embeddings, user_embeddings.unsqueeze(1), post_embeddings.unsqueeze(1)], dim=1)
            torch.save(data, DATASET_SAVING_DIR + "repost_{}.pt".format(self._graph._cur_tag))
            print(data.shape)
            # Comment
            print(f'Building comment_{self._graph._cur_tag}.pt')
            uid, pid = self._graph.graph.edges(etype='comment_on')
            musk = self._graph.graph.edges['comment_on'].data['tag'] == self._graph._cur_tag
            uid = uid[musk]
            pid = pid[musk]
            user_history_interacted_post_embeddings = self._graph.graph.nodes['user'].data['history_interacted'][uid]
            user_embeddings = self._graph.graph.nodes['user'].data['embedding'][uid]
            post_embeddings = self._graph.graph.nodes['post'].data['embedding'][pid]
            data = torch.cat([user_history_interacted_post_embeddings, user_embeddings.unsqueeze(1), post_embeddings.unsqueeze(1)], dim=1)
            torch.save(data, DATASET_SAVING_DIR + "comment_{}.pt".format(self._graph._cur_tag))
            print(data.shape)
            # Like
            print(f'Building like_{self._graph._cur_tag}.pt')
            uid, pid = self._graph.graph.edges(etype='like')
            musk = self._graph.graph.edges['like'].data['tag'] == self._graph._cur_tag
            uid = uid[musk]
            pid = pid[musk]
            user_history_interacted_post_embeddings = self._graph.graph.nodes['user'].data['history_interacted'][uid]
            user_embeddings = self._graph.graph.nodes['user'].data['embedding'][uid]
            post_embeddings = self._graph.graph.nodes['post'].data['embedding'][pid]
            data = torch.cat([user_history_interacted_post_embeddings, user_embeddings.unsqueeze(1), post_embeddings.unsqueeze(1)], dim=1)
            torch.save(data, DATASET_SAVING_DIR + "like_{}.pt".format(self._graph._cur_tag))
            print(data.shape)
            
            self._graph.update_tag()
            if i < 6:  # skip the last round, as it's not needed
                self.update_state()

    def collect_action_density_this_round(self):
        self.msg_sender.mailbox["tag"] = self._graph._cur_tag
        self.collect_action_density_this_round_for_etype('create_post_r', 'post')  
        self.collect_action_density_this_round_for_etype('repost_on_r', 'repost')
        self.collect_action_density_this_round_for_etype('comment_on_r', 'comment')
        self.collect_action_density_this_round_for_etype('like_r', 'like')

    def collect_action_density_this_round_for_etype(self, etype, action_name):
        id_edge_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_with_tag, etype=etype).int()
        self._graph.graph.send_and_recv(
            id_edge_within_window,
            self.msg_count_action,
            self.reduce_count_action,
            etype=("post", etype, "user"),
        )
        self._graph.graph.nodes["user"].data["{}_count_this_round".format(action_name)] = self._graph.graph.nodes["user"].data["count_action"]
        del self._graph.graph.nodes["user"].data["count_action"]  # release cache

    def msg_count_action(self, edges):
        return {
            "edge_tag": edges.data["tag"], 
        }

    def reduce_count_action(self, nodes):
        return {
            "count_action": torch.ones(nodes.mailbox["edge_tag"].shape[0]) * nodes.mailbox["edge_tag"].shape[1],
        }
