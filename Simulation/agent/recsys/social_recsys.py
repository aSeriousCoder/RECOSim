import torch
import numpy as np
import pandas as pd
import dgl
from tqdm import tqdm
from Simulation.agent.recsys.base_recsys import BaseRecSys


class SocialRecSys(BaseRecSys):
    def __init__(self, config, logger, graph):
        self._config = config
        self._logger = logger
        self._graph = graph
        self._post_rec_list_length = config.post_rec_list_length
        self._friend_rec_list_length = config.friend_rec_list_length
        self.msg_sender = MsgSender()
        self.msg_sender.mailbox['post_rec_list_length'] = config.post_rec_list_length
        self.msg_sender.mailbox['friend_rec_list_length'] = config.friend_rec_list_length
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
        '''
        关注的人发布/交互过的帖子
        post repost / like comment_on
        '''
        self._logger.debug("Recommending News from Social Networks ...")
        # window中采样
        self.msg_sender.mailbox['begin_day'] = self._graph._cur_tag - self._config.window_size
        self.msg_sender.mailbox['end_day'] = self._graph._cur_tag
        id_post_up_to_now = self._graph.graph.filter_edges(self.msg_sender.interactions_in_window, etype='create_post').int()
        self.msg_sender.mailbox['id_post_up_to_now'] = id_post_up_to_now
        candidate_mat = torch.randint(0, len(id_post_up_to_now), [self._graph.graph.num_nodes('user'), self._post_rec_list_length * 10])
        candidate_mat = torch.stack([id_post_up_to_now[candidate_list.long()] for candidate_list in candidate_mat], dim=0)  # 随机打底
        
        sub_g = self._graph.graph.edge_type_subgraph(['follow', 'follow_r'])

        history_begin_day = self._graph._cur_tag - self._config.window_size
        history_end_day = self._graph._cur_tag
        create_post_links = self._graph.graph.edges(etype="create_post")
        create_post_links_in_window_mask = (
            self._graph.graph.edges["create_post"].data["tag"] >= history_begin_day
        ) & (self._graph.graph.edges["create_post"].data["tag"] < history_end_day)
        create_post_df = pd.DataFrame(
            {
                "user": create_post_links[0][create_post_links_in_window_mask].numpy(),
                "post": create_post_links[1][create_post_links_in_window_mask].numpy(),
            }
        )
        like_links = self._graph.graph.edges(etype="like")
        like_links_in_window_mask = (
            self._graph.graph.edges["like"].data["tag"] >= history_begin_day
        ) & (self._graph.graph.edges["like"].data["tag"] < history_end_day)
        like_df = pd.DataFrame(
            {
                "user": like_links[0][like_links_in_window_mask].numpy(),
                "post": like_links[1][like_links_in_window_mask].numpy(),
            }
        )
        repost_on_links = self._graph.graph.edges(etype="repost_on")
        repost_on_links_in_window_mask = (
            self._graph.graph.edges["repost_on"].data["tag"] >= history_begin_day
        ) & (self._graph.graph.edges["repost_on"].data["tag"] < history_end_day)
        repost_on_df = pd.DataFrame(
            {
                "user": repost_on_links[0][repost_on_links_in_window_mask].numpy(),
                "post": repost_on_links[1][repost_on_links_in_window_mask].numpy(),
            }
        )
        # comment_on_links = self._graph.graph.edges(etype="comment_on")
        # comment_on_links_in_window_mask = (
        #     self._graph.graph.edges["comment_on"].data["tag"] >= history_begin_day
        # ) & (self._graph.graph.edges["comment_on"].data["tag"] < history_end_day)
        # comment_on_df = pd.DataFrame(
        #     {
        #         "user": comment_on_links[0][comment_on_links_in_window_mask].numpy(),
        #         "post": comment_on_links[1][comment_on_links_in_window_mask].numpy(),
        #     }
        # )
        like_augment_links = like_df.join(
            create_post_df.set_index("post"),
            on="post",
            how="inner",
            lsuffix="_src",
            rsuffix="_dst",
        )
        repost_augment_links = repost_on_df.join(
            create_post_df.set_index("post"),
            on="post",
            how="inner",
            lsuffix="_src",
            rsuffix="_dst",
        )
        # comment_augment_links = comment_on_df.join(
        #     create_post_df.set_index("post"),
        #     on="post",
        #     how="inner",
        #     lsuffix="_src",
        #     rsuffix="_dst",
        # )
        augment_links = pd.concat(
            [
                like_augment_links[["user_src", "user_dst"]],
                repost_augment_links[["user_src", "user_dst"]],
                # comment_augment_links[["user_src", "user_dst"]],
            ]
        ).values
        augment_links = [[int(a[0]), int(a[1])] for a in augment_links if a[0] != a[1]]
        augment_links = np.unique(np.array(augment_links), axis=0)
        sub_g.add_edges(u=augment_links[:,0], v=augment_links[:, 1], etype='follow')
        sub_g.add_edges(u=augment_links[:,1], v=augment_links[:, 0], etype='follow_r')

        follow_record = self._graph.graph.adj('follow').indices().T
        follow_record_df_first_hop = pd.DataFrame({
            'src_user': follow_record[:, 0],
            'one_hop_user': follow_record[:, 1]
        })
        follow_record_df_second_hop = pd.DataFrame({
            'one_hop_user': follow_record[:, 0],
            'two_hop_user': follow_record[:, 1]
        })
        follow_record_df_two_hop = follow_record_df_first_hop.join(
            follow_record_df_second_hop.set_index("one_hop_user"),
            on="one_hop_user",
            how="inner",
            lsuffix="_src",
            rsuffix="_dst",
        )
        two_hop_follow_relations = follow_record_df_two_hop[['src_user', 'two_hop_user']].values
        sub_g.add_edges(u=two_hop_follow_relations[:,0], v=two_hop_follow_relations[:, 1], etype='follow')
        sub_g.add_edges(u=two_hop_follow_relations[:,1], v=two_hop_follow_relations[:, 0], etype='follow_r')

        follow_record_df_first_hop = pd.DataFrame({
            'src_user': follow_record[:, 0],
            'user_src': follow_record[:, 1]
        })
        augment_links = pd.concat(
            [
                like_augment_links[["user_src", "user_dst"]],
                repost_augment_links[["user_src", "user_dst"]],
                # comment_augment_links[["user_src", "user_dst"]],
            ]
        )
        following_user_also_intered_with = follow_record_df_first_hop.join(
            augment_links.set_index("user_src"),
            on="user_src",
            how="inner",
            lsuffix="_src",
            rsuffix="_dst",
        )
        following_user_also_intered_with_relations = following_user_also_intered_with[['src_user', 'user_dst']].values
        sub_g.add_edges(u=following_user_also_intered_with_relations[:,0], v=following_user_also_intered_with_relations[:, 1], etype='follow')
        sub_g.add_edges(u=following_user_also_intered_with_relations[:,1], v=following_user_also_intered_with_relations[:, 0], etype='follow_r')

        sub_g.update_all(
            self.msg_sender.msg_social_recall_post,
            self.msg_sender.reduce_social_recall_post,
            etype=("user", "follow_r", "user"),
        )
        social_recall_post = sub_g.nodes["user"].data["social_recall_post"]
        social_recall_post_length = sub_g.nodes["user"].data["social_recall_post_length"]
        candidate_mat_in_pair = []
        for i in tqdm(range(self._graph.graph.num_nodes('user')), disable=not self._config.show_progress):
            recommendation_length = social_recall_post_length[i]
            recommended_post = social_recall_post[i][:recommendation_length]
            candidate_mat_in_pair.extend([[i, post] for post in recommended_post])
        candidate_mat_in_pair = torch.tensor(candidate_mat_in_pair).int()

        filter_out_mat_1 = self._graph.graph.has_edges_between(candidate_mat_in_pair[:,0].int(), candidate_mat_in_pair[:,1].int(), ('user', 'post_rec', 'post'))
        filter_out_mat_2 = self._graph.graph.has_edges_between(candidate_mat_in_pair[:,0].int(), candidate_mat_in_pair[:,1].int(), ('user', 'create_post', 'post'))
        in_init_musk = candidate_mat_in_pair[:, 1] < self.interaction_sub_g.num_nodes('post')  # init之后的交互一定会被post_rec所覆盖
        filter_out_mat_3 = torch.zeros(candidate_mat_in_pair.shape[0]).bool()
        filter_out_mat_3[in_init_musk] = self.interaction_sub_g.has_edges_between(candidate_mat_in_pair[in_init_musk][:,0].int(), candidate_mat_in_pair[in_init_musk][:,1].int(), ('user', 'like', 'post'))
        filter_out_mat_4 = torch.zeros(candidate_mat_in_pair.shape[0]).bool()
        filter_out_mat_4[in_init_musk] = self.interaction_sub_g.has_edges_between(candidate_mat_in_pair[in_init_musk][:,0].int(), candidate_mat_in_pair[in_init_musk][:,1].int(), ('user', 'comment_on', 'post'))
        filter_out_mat_5 = torch.zeros(candidate_mat_in_pair.shape[0]).bool()
        filter_out_mat_5[in_init_musk] = self.interaction_sub_g.has_edges_between(candidate_mat_in_pair[in_init_musk][:,0].int(), candidate_mat_in_pair[in_init_musk][:,1].int(), ('user', 'repost_on', 'post'))
        filter_out_mat = filter_out_mat_1 | filter_out_mat_2 | filter_out_mat_3 | filter_out_mat_4 | filter_out_mat_5

        candidate_mat_in_pair = candidate_mat_in_pair[~filter_out_mat]
        selected_mat_in_pair = []
        for i in tqdm(range(self._graph.graph.num_nodes('user')), disable=not self._config.show_progress):
            user_recalled_candidates = candidate_mat_in_pair[candidate_mat_in_pair[:,0] == i][:,1]
            if len(user_recalled_candidates) > self._post_rec_list_length:
                random_indices = torch.randperm(user_recalled_candidates.size(0))
                shuffled_user_recalled_candidates = user_recalled_candidates[random_indices]
                selected_mat_in_pair.extend([[i, post] for post in shuffled_user_recalled_candidates[:self._post_rec_list_length]])
            else:
                selected_mat_in_pair.extend([[i, post] for post in user_recalled_candidates])
        selected_mat_in_pair = torch.tensor(selected_mat_in_pair).int()
        
        return selected_mat_in_pair

    def recommend_friends(self):
        self._logger.debug("Recommend Friends from Social Networks ...")
        follow_record = self._graph.graph.adj('follow').indices().T
        follow_record_df_first_hop = pd.DataFrame({
            'src_user': follow_record[:, 0],
            'one_hop_user': follow_record[:, 1]
        })
        follow_record_df_second_hop = pd.DataFrame({
            'one_hop_user': follow_record[:, 0],
            'two_hop_user': follow_record[:, 1]
        })
        follow_record_df_two_hop = follow_record_df_first_hop.join(
            follow_record_df_second_hop.set_index("one_hop_user"),
            on="one_hop_user",
            how="inner",
            lsuffix="_src",
            rsuffix="_dst",
        )
        two_hop_follow_relations = follow_record_df_two_hop[['src_user', 'two_hop_user']].values
        two_hop_follow_graph = dgl.heterograph(
            {('user', 'follow_r', 'user'): (torch.from_numpy(two_hop_follow_relations)[:,1].int(), torch.from_numpy(two_hop_follow_relations)[:,0].int())}, 
            num_nodes_dict={'user': self._graph.graph.num_nodes('user')}
        )  # reverse prop
        two_hop_follow_graph.nodes['user'].data['_ID'] = torch.tensor(list(range(two_hop_follow_graph.num_nodes('user')))).long()
        self.msg_sender.mailbox['num_users'] = self._graph.graph.num_nodes('user')
        two_hop_follow_graph.update_all(
            self.msg_sender.msg_social_recall_friend,
            self.msg_sender.reduce_social_recall_friend,
            etype=("user", "follow_r", "user"),
        )
        social_recall_friend = two_hop_follow_graph.nodes["user"].data["social_recall_friend"]
        social_recall_friend_length = two_hop_follow_graph.nodes["user"].data["social_recall_friend_length"]
        candidate_mat_in_pair = []
        for i in tqdm(range(self._graph.graph.num_nodes('user')), disable=not self._config.show_progress):
            recommendation_length = social_recall_friend_length[i]
            recommended_friend = social_recall_friend[i][:recommendation_length]
            candidate_mat_in_pair.extend([[i, friend] for friend in recommended_friend])
        candidate_mat_in_pair = torch.tensor(candidate_mat_in_pair).int()

        filter_out_mat_1 = self._graph.graph.has_edges_between(candidate_mat_in_pair[:,0].int(), candidate_mat_in_pair[:,1].int(), ('user', 'friend_rec', 'user'))
        filter_out_mat_2 = self.follow_sub_g.has_edges_between(candidate_mat_in_pair[:,0].int(), candidate_mat_in_pair[:,1].int(), ('user', 'follow', 'user'))  # init
        filter_out_mat = filter_out_mat_1 | filter_out_mat_2

        candidate_mat_in_pair = candidate_mat_in_pair[~filter_out_mat]
        selected_mat_in_pair = []
        for i in tqdm(range(self._graph.graph.num_nodes('user')), disable=not self._config.show_progress):
            user_recalled_candidates = candidate_mat_in_pair[candidate_mat_in_pair[:,0] == i][:,1]
            if len(user_recalled_candidates) > self._friend_rec_list_length:
                random_indices = torch.randperm(user_recalled_candidates.size(0))
                shuffled_user_recalled_candidates = user_recalled_candidates[random_indices]
                selected_mat_in_pair.extend([[i, post] for post in shuffled_user_recalled_candidates[:self._friend_rec_list_length]])
            else:
                selected_mat_in_pair.extend([[i, post] for post in user_recalled_candidates])
        selected_mat_in_pair = torch.tensor(selected_mat_in_pair).int()

        return selected_mat_in_pair


class MsgSender:
    def __init__(self):
        self.mailbox = {}

    def interactions_in_window(self, edges):
        begin_day = self.mailbox["begin_day"]
        end_day = self.mailbox["end_day"]
        edge_tag = edges.data['tag']
        return (edge_tag >= begin_day) & (edge_tag < end_day)
    
    def msg_social_recall_post(self, edges):
        return {
            'tmpvar_history_post_ID': edges.src['history_post_ID'],
            'tmpvar_history_repost_ID': edges.src['history_repost_ID'],
            'tmpvar_history_comment_ID': edges.src['history_comment_ID'],
            'tmpvar_history_like_ID': edges.src['history_like_ID'],
        }

    def reduce_social_recall_post(self, nodes):
        post_rec_list_length = self.mailbox["post_rec_list_length"]
        id_post_up_to_now = self.mailbox["id_post_up_to_now"]
        tmpvar_history_post_ID = nodes.mailbox['tmpvar_history_post_ID']
        tmpvar_history_repost_ID = nodes.mailbox['tmpvar_history_repost_ID']
        tmpvar_history_comment_ID = nodes.mailbox['tmpvar_history_comment_ID']
        tmpvar_history_like_ID = nodes.mailbox['tmpvar_history_like_ID']
        interact_record = torch.cat([tmpvar_history_post_ID, tmpvar_history_repost_ID, tmpvar_history_comment_ID, tmpvar_history_like_ID], dim=-1).reshape(len(nodes), -1)
        maxium_recall_length = 10 * post_rec_list_length
        candidate_from_interact = torch.zeros([len(nodes), maxium_recall_length]).int()
        num_candidate = torch.zeros([len(nodes)]).int()
        for i, record in enumerate(interact_record):
            record = record.unique()
            if record[0] == 0:
                record = record[1:]  # 0 is the padding for empty history
            candidate_length = min(len(record), maxium_recall_length)
            candidate_from_interact[i, :candidate_length] = record[-candidate_length:]  # we tend to recommend newer posts
            num_candidate[i] = candidate_length
        return {
            'social_recall_post': candidate_from_interact,
            'social_recall_post_length': num_candidate,
        }

    def msg_social_recall_friend(self, edges):
        return {'tmpvar_ID': edges.src['_ID']}

    def reduce_social_recall_friend(self, nodes):
        friend_rec_list_length = self.mailbox["friend_rec_list_length"]
        num_users = self.mailbox["num_users"]
        tmpvar_ID = nodes.mailbox['tmpvar_ID']
        maxium_recall_length = 10 * friend_rec_list_length
        candidate_friend = torch.zeros([len(nodes), maxium_recall_length]).int()
        num_candidate = torch.zeros([len(nodes)]).int()
        for i, friend_IDs in enumerate(tmpvar_ID):
            friend_IDs = friend_IDs.unique()
            if len(friend_IDs) <= maxium_recall_length:
                candidate_friend[i, :len(friend_IDs)] = friend_IDs
                num_candidate[i] = len(friend_IDs)
            else:
                random_indices = torch.randperm(friend_IDs.size(0))
                shuffled_friend_IDs = friend_IDs[random_indices]
                candidate_friend[i, :maxium_recall_length] = shuffled_friend_IDs[:maxium_recall_length]
                num_candidate[i] = maxium_recall_length
        return {
            'social_recall_friend': candidate_friend,
            'social_recall_friend_length': num_candidate,
        }
