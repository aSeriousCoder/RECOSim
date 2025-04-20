import dgl
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from Simulation.agent.user.base_user import BaseUser


class AwesomeUser(BaseUser):
    def __init__(self, config, logger, graph, user_action_service, metrics_service):
        self._config = config
        self._logger = logger
        self._graph = graph
        self._user_action_service = user_action_service
        self._metrics_service = metrics_service
        self._post_rec_list_length = config.post_rec_list_length
        self._friend_rec_list_length = config.friend_rec_list_length
        self._window_size = config.window_size
        self._top_k_pop = config.top_k_pop
        self._padding = config.padding
        self._embedding_dim = config.embedding_dim
        self._simu_batch_size = config.simu_batch_size
        self._max_post_num_per_round = config.max_post_num_per_round
        self._history_length = config.history_length
        self._gmm_topic_num = config.gmm_topic_num

        self.msg_sender = MsgSender(graph)
        self.msg_sender._window_size = config.window_size
        self.msg_sender._top_k_pop = config.top_k_pop
        self.msg_sender._padding = config.padding
        self.msg_sender._embedding_dim = config.embedding_dim
        self.msg_sender._simu_batch_size = config.simu_batch_size
        self.msg_sender._max_post_num_per_round = config.max_post_num_per_round
        self.msg_sender._history_length = config.history_length
        self.msg_sender._gmm_topic_num = config.gmm_topic_num

        if not self._config.continue_simulation:
            self.update_sampling_and_embedding(is_init=True)
            if not self._config.build_dataset: 
                self.init_follow_threshold()
        
        if self._config.build_dataset:
            self.update_state()

    def update_sampling_and_embedding(self, is_init):
        """
        To init:
        1. history_interacted & history_interacted_ID
        2. embedding
        3. topic_engage
        """
        if is_init:
            msg_func = self.msg_sender.msg_sample_user_history_interactions_by_time
            reduce_func = self.msg_sender.reduce_sample_user_history_interactions_by_time
        else:
            msg_func = self.msg_sender.msg_sample_user_history_interactions_by_through_pass
            reduce_func = self.msg_sender.reduce_sample_user_history_interactions_by_through_pass
            self._graph.graph.edges['create_repost_r'].data['through_pass'] = self._graph.graph.edges['repost_on_r'].data['through_pass']
            self._graph.graph.edges['create_comment_r'].data['through_pass'] = self._graph.graph.edges['comment_on_r'].data['through_pass']

        self._logger.debug("Init User History, Embedding and Topic Engagement ...")
        cur_tag = self._graph._cur_tag
        history_window_beginning = self._graph._history_window_beginning
        self.msg_sender.mailbox["window_begin"] = history_window_beginning
        self.msg_sender.mailbox["window_end"] = cur_tag

        eids_create_post_r_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype='create_post_r').int()
        eids_create_repost_r_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype='create_repost_r').int()
        eids_create_comment_r_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype='create_comment_r').int()
        eids_like_r_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype='like_r').int()

        self.msg_sender.mailbox["cur_processing_action"] = 'post'
        self._graph.graph.send_and_recv(eids_create_post_r_within_window, msg_func, reduce_func, etype=('post', 'create_post_r', 'user'))
        self.msg_sender.mailbox["cur_processing_action"] = 'repost'
        self._graph.graph.send_and_recv(eids_create_repost_r_within_window, msg_func, reduce_func, etype=('repost', 'create_repost_r', 'user'))
        self.msg_sender.mailbox["cur_processing_action"] = 'comment'
        self._graph.graph.send_and_recv(eids_create_comment_r_within_window, msg_func, reduce_func, etype=('comment', 'create_comment_r', 'user'))
        self.msg_sender.mailbox["cur_processing_action"] = 'like'
        self._graph.graph.send_and_recv(eids_like_r_within_window, msg_func, reduce_func, etype=('post', 'like_r', 'user'))

        self._graph.graph.nodes['user'].data['embedding'] = torch.concat([
            self._graph.graph.nodes['user'].data['history_post'],
            self._graph.graph.nodes['user'].data['history_repost'],
            self._graph.graph.nodes['user'].data['history_comment'],
            self._graph.graph.nodes['user'].data['history_like'],
        ], dim=1).mean(dim=1)
        self._graph.graph.nodes['user'].data['history_interacted'] = torch.concat([
            self._graph.graph.nodes['user'].data['history_post'],
            self._graph.graph.nodes['user'].data['history_repost'],
            self._graph.graph.nodes['user'].data['history_comment'],
            self._graph.graph.nodes['user'].data['history_like'],
        ], dim=1)
        self._graph.graph.nodes['user'].data['history_interacted_ID'] = torch.concat([
            self._graph.graph.nodes['user'].data['history_post_ID'],
            self._graph.graph.nodes['user'].data['history_repost_ID'],
            self._graph.graph.nodes['user'].data['history_comment_ID'],
            self._graph.graph.nodes['user'].data['history_like_ID'],
        ], dim=1)
        self._graph.graph.nodes['user'].data['topic_engage'] = torch.concat([
            self._graph.graph.nodes['user'].data['topic_engage_post'],
            self._graph.graph.nodes['user'].data['topic_engage_repost'],
            self._graph.graph.nodes['user'].data['topic_engage_comment'],
            self._graph.graph.nodes['user'].data['topic_engage_like'],
        ], dim=1)

        self._metrics_service.append('user_embedding', self._graph.graph.nodes['user'].data['embedding'])
        self._metrics_service.append('user_topic_engage', self._graph.graph.nodes['user'].data['topic_engage'])
    
    # Trained Model Required
    def init_follow_threshold(self):
        # Only for Simulation
        # When use as DatasetBuilder, this is not used
        self._logger.debug("Init Follow Threshold ...")
        batch_size = self._simu_batch_size
        following_relations = self._graph.graph.adj('follow').indices()
        src_user_history_embedding = self._graph.graph.nodes["user"].data['history_interacted'][following_relations[0]]
        dst_user_history_embedding = self._graph.graph.nodes["user"].data['history_interacted'][following_relations[1]]
        src_user_embedding = self._graph.graph.nodes["user"].data['embedding'][following_relations[0]]
        dst_user_embedding = self._graph.graph.nodes["user"].data['embedding'][following_relations[1]]
        num_batch = int(following_relations.shape[1]/batch_size) + 1
        follow_scores = []
        for i in tqdm(range(num_batch), disable=not self._config.show_progress):
            begin = i * batch_size
            end = (i + 1) * batch_size
            batch = (
                src_user_history_embedding[begin:end].to(self._config.device),
                src_user_embedding[begin:end].to(self._config.device),
                dst_user_history_embedding[begin:end].to(self._config.device), 
                dst_user_embedding[begin:end].to(self._config.device),
            )
            follow_score = self._user_action_service.score("follow", batch).detach().cpu()
            follow_scores.append(follow_score)
        follow_scores = torch.cat(follow_scores, dim=0)
        default_follow_threshold = torch.quantile(follow_scores, 0.8)
        default_unfollow_threshold = torch.quantile(follow_scores, 0.2)
        user_follow_scores = {i:[] for i in range(self._graph.graph.num_nodes('user'))}
        for i in range(following_relations.shape[1]):
            user_id = int(following_relations[0][i])
            score = float(follow_scores[i])
            user_follow_scores[user_id].append(score)
        user_follow_threshold = torch.ones(self._graph.graph.num_nodes('user')) * default_follow_threshold
        user_unfollow_threshold = torch.ones(self._graph.graph.num_nodes('user')) * default_unfollow_threshold
        for user_id, scores in user_follow_scores.items():
            if len(scores) > 10:
                user_follow_threshold[user_id] = torch.quantile(torch.tensor(scores), 0.8)
                user_unfollow_threshold[user_id] = torch.quantile(torch.tensor(scores), 0.2)
        self._graph.graph.nodes["user"].data["follow_threshold"] = user_follow_threshold
        self._graph.graph.nodes["user"].data["unfollow_threshold"] = user_unfollow_threshold

    def update_state(self):
        """
        Using data in active window (cur_tag - window_size, cur_tag) to update the state of the users:
        1. Update Usage and Gratification State
        2. Update User History Interaction based on UG-state
        3. Update User Embedding and GMM Signal
        4. Update User Action Density
        """
        self._logger.debug("Updating User State ...")
        cur_tag = self._graph._cur_tag
        history_window_beginning = self._graph._history_window_beginning
        self.msg_sender.mailbox["tag"] = cur_tag
        self.msg_sender.mailbox["window_begin"] = history_window_beginning
        self.msg_sender.mailbox["window_end"] = cur_tag

        # Update Usage
        self._logger.debug("Updating Usage ...")
        self.update_usage()

        # Update Gratification
        self.update_gratification()
        if "last_ugt_state" not in self._graph.graph.nodes["user"].data:
            self._graph.graph.nodes["user"].data["last_ugt_state"] = self._graph.graph.nodes["user"].data["ugt_state"]

        # Update User Embedding and GMM Signal
        self.update_sampling_and_embedding(is_init=False)

        # Update User Action Density
        if not self._config.build_dataset: 
            self.update_action_density_and_threshold()

    def update_usage(self):
        self.update_usage_for_etype('create_post_r', 'post')  
        self.update_usage_for_etype('repost_on_r', 'repost')
        self.update_usage_for_etype('comment_on_r', 'comment')
        self.update_usage_for_etype('like_r', 'like')

    def update_usage_for_etype(self, etype, action_name):
        id_edge_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype=etype).int()
        self._graph.graph.send_and_recv(
            id_edge_within_window,
            self.msg_sender.msg_count_usage,
            self.msg_sender.reduce_count_usage,
            etype=("post", etype, "user"),
        )
        self._graph.graph.nodes["user"].data["{}_count".format(action_name)] = self._graph.graph.nodes["user"].data["count_usage"]
        del self._graph.graph.nodes["user"].data["count_usage"]  # release cache

    def update_gratification(self):
        self._logger.debug("Updating Gratification ...")

        self._graph.graph.edges["create_post_r"].data["through_pass"] = torch.zeros(self._graph.graph.num_edges("create_post_r"))
        self._graph.graph.edges["repost_on_r"].data["through_pass"] = torch.zeros(self._graph.graph.num_edges("repost_on_r"))
        self._graph.graph.edges["comment_on_r"].data["through_pass"] = torch.zeros(self._graph.graph.num_edges("comment_on_r"))
        self._graph.graph.edges["like_r"].data["through_pass"] = torch.zeros(self._graph.graph.num_edges("like_r"))

        eids_create_post_r_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype='create_post_r').int()
        eids_create_repost_r_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype='create_repost_r').int()
        eids_create_comment_r_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype='create_comment_r').int()
        eids_like_r_within_window = self._graph.graph.filter_edges(self.msg_sender.edges_within_window, etype='like_r').int()

        # Seeking Info - 1 Something Interesting - Avg & Max Interacted Post Similarity
        self._logger.debug("Collecting Seeking Info - 1 Something Interesting")
        # Seeking Info - 2 Something Useful - Avg & Max Interacted Post Popularity
        self._logger.debug("Collecting Seeking Info - 2 Something Useful")
        # 不是用户自己发的，而是用户互动过的
        self._graph.graph.nodes["post"].data["popularity"] = (
            self._graph.graph.in_degrees(etype="like") + self._graph.graph.in_degrees(etype="repost_of") + self._graph.graph.in_degrees(etype="comment_of")
        ).float()
        # user <-[like_r]- post
        self.msg_sender.mailbox["cur_rel"] = "like"
        self.msg_sender.mailbox["cur_etype"] = "like_r"
        self._graph.graph.send_and_recv(
            eids_like_r_within_window,
            self.msg_sender.msg_interaction_cossim_and_popularity,
            self.msg_sender.reduce_interaction_cossim_and_popularity,
            etype=("post", "like_r", "user"),
        )
        # user <-[create_repost_r]- repost <-[repost_of_r]- post
        self.msg_sender.mailbox["cur_rel"] = "repost"
        self.msg_sender.mailbox["cur_etype"] = "repost_on_r"
        self._graph.graph.update_all(
            self.msg_sender.msg_repeat_emb_and_popularity,
            self.msg_sender.reduce_repeat_emb_and_popularity,
            etype=("post", "repost_of_r", "repost"),
        )
        self._graph.graph.send_and_recv(
            eids_create_repost_r_within_window,
            self.msg_sender.msg_interaction_cossim_and_popularity_from_repeat,
            self.msg_sender.reduce_interaction_cossim_and_popularity,
            etype=("repost", "create_repost_r", "user"),
        )
        del self._graph.graph.nodes["repost"].data["repeat_embedding"]  # release cache
        del self._graph.graph.nodes["repost"].data["repeat_popularity"]  # release cache
        # user <-[create_comment_r]- comment <-[comment_of_r]- post
        self.msg_sender.mailbox["cur_rel"] = "comment"
        self.msg_sender.mailbox["cur_etype"] = "comment_on_r"
        self._graph.graph.update_all(
            self.msg_sender.msg_repeat_emb_and_popularity,
            self.msg_sender.reduce_repeat_emb_and_popularity,
            etype=("post", "comment_of_r", "comment"),
        )
        self._graph.graph.send_and_recv(
            eids_create_comment_r_within_window,
            self.msg_sender.msg_interaction_cossim_and_popularity_from_repeat,
            self.msg_sender.reduce_interaction_cossim_and_popularity,
            etype=("comment", "create_comment_r", "user"),
        )
        del self._graph.graph.nodes["comment"].data["repeat_embedding"]  # release cache
        del self._graph.graph.nodes["comment"].data["repeat_popularity"]  # release cache

        # Seeking Info - 3 Something New - Avg Interacted Post Time Interval
        self._logger.debug("Collecting Seeking Info - 3 Something New")
        time_interval = torch.Tensor(list(range(self._window_size, 0, -1)))
        # Post
        daily_usage_post_in_window = self._graph.graph.nodes["user"].data["post_count"]
        avg_post_time_interval = (daily_usage_post_in_window * time_interval).sum(1) / (
            daily_usage_post_in_window.sum(1) + 1e-8
        )  # 没交互过是0，应该设置为self._window_size
        avg_post_time_interval += (avg_post_time_interval == 0).int() * self._window_size
        self._graph.graph.nodes["user"].data["avg_post_time_interval"] = avg_post_time_interval
        # Repost
        daily_usage_repost_in_window = self._graph.graph.nodes["user"].data["repost_count"]
        avg_repost_time_interval = (daily_usage_repost_in_window * time_interval).sum(1) / (
            daily_usage_repost_in_window.sum(1) + 1e-8
        )  # 没交互过是0，应该设置为self._window_size
        avg_repost_time_interval += (avg_repost_time_interval == 0).int() * self._window_size
        self._graph.graph.nodes["user"].data["avg_repost_time_interval"] = avg_repost_time_interval
        # Comment
        daily_usage_comment_in_window = self._graph.graph.nodes["user"].data["comment_count"]
        avg_comment_time_interval = (daily_usage_comment_in_window * time_interval).sum(1) / (
            daily_usage_comment_in_window.sum(1) + 1e-8
        )  # 没交互过是0，应该设置为self._window_size
        avg_comment_time_interval += (
            avg_comment_time_interval == 0
        ).int() * self._window_size
        self._graph.graph.nodes["user"].data["avg_comment_time_interval"] = avg_comment_time_interval
        # Like
        daily_usage_like_in_window = self._graph.graph.nodes["user"].data["like_count"]
        avg_like_time_interval = (daily_usage_like_in_window * time_interval).sum(1) / (
            daily_usage_like_in_window.sum(1) + 1e-8
        )  # 没交互过是0，应该设置为self._window_size
        avg_like_time_interval += (avg_like_time_interval == 0).int() * self._window_size
        self._graph.graph.nodes["user"].data["avg_like_time_interval"] = avg_like_time_interval

        # Sharing Info - 1 Self Expression - Avg & Max Self Post Similarity
        self._logger.debug("Collecting Sharing Info - 1 Self Expression")
        # user <-[create_post_r]- post
        self.msg_sender.mailbox["cur_rel"] = "post"
        self.msg_sender.mailbox["cur_etype"] = "create_post_r"
        self._graph.graph.send_and_recv(
            eids_create_post_r_within_window,
            self.msg_sender.msg_interaction_cossim,
            self.msg_sender.reduce_interaction_cossim,
            etype=("post", "create_post_r", "user"),
        )
        # user <- [create_repost_r] - repost
        self.msg_sender.mailbox["cur_rel"] = "repost"
        self.msg_sender.mailbox["cur_etype"] = "repost_on_r"
        self._graph.graph.send_and_recv(
            eids_create_repost_r_within_window,
            self.msg_sender.msg_interaction_cossim,
            self.msg_sender.reduce_interaction_cossim,
            etype=("repost", "create_repost_r", "user"),
        )
        # user <- [create_comment_r] - comment
        self.msg_sender.mailbox["cur_rel"] = "comment"
        self.msg_sender.mailbox["cur_etype"] = "comment_on_r"
        self._graph.graph.send_and_recv(
            eids_create_comment_r_within_window,
            self.msg_sender.msg_interaction_cossim,
            self.msg_sender.reduce_interaction_cossim,
            etype=("comment", "create_comment_r", "user"),
        )

        # Sharing Info - 2 Provide to Others - Avg & Max Cos Similarity of MyFollowers-MyPosts
        self._logger.debug("Collecting Sharing Info - 2 Provide to Others")
        # user <-[follow]- user
        self._graph.graph.update_all(
            self.msg_sender.msg_avg_follower_embedding,
            self.msg_sender.reduce_avg_follower_embedding,
            etype=("user", "follow", "user"),
        )
        # user <-[create_post_r]- post
        self.msg_sender.mailbox["cur_etype"] = "create_post_r"
        self._graph.graph.send_and_recv(
            eids_create_post_r_within_window,
            self.msg_sender.msg_avg_posted_embedding,
            self.msg_sender.reduce_avg_cross_follower_and_post_similarity,
            etype=("post", "create_post_r", "user"),
        )
        # self._graph.graph.nodes["user"].data["avg_cross_follower_and_post_similarity"] = F.cosine_similarity(
        #     self._graph.graph.nodes["user"].data["tmpvar_avg_follower_embedding"],
        #     self._graph.graph.nodes["user"].data["tmpvar_avg_post_embedding"],
        #     dim=-1,
        # )  # have merged into reduce_avg_cross_follower_and_post_similarity
        del self._graph.graph.nodes["user"].data["tmpvar_avg_follower_embedding"]  # release cache
        # self._graph.graph.nodes["user"].data["tmpvar_avg_post_embedding"] 后面Social Interaction - Value Sharing还会用，暂不释放

        # Sharing Info - 3 Contribute to Community - Avg & Max Cos Similarity of CommunityInterest-MyPosts
        self._logger.debug("Collecting Sharing Info - 3 Contribute to Community")
        # Community Type 1: Relationship Based
        sub_g_rel = dgl.to_homogeneous(
            self._graph.graph.edge_type_subgraph(["follow"])
        )  # It also contains all nodes of incident type
        nx_g_rel = dgl.to_networkx(sub_g_rel)
        partition_rel = nx.community.louvain_communities(nx_g_rel)
        self._graph.graph.nodes["user"].data["partition_rel"] = torch.zeros(
            self._graph.graph.num_nodes("user")
        ).int()
        for i, p in enumerate(partition_rel):
            self._graph.graph.nodes["user"].data["partition_rel"][list(p)] = i
        community_centers_rel = []
        for p in partition_rel:
            community_centers_rel.append(
                self._graph.graph.nodes["user"]
                .data["embedding"][torch.Tensor(list(p)).long()]
                .mean(0)
            )
        self.msg_sender.mailbox["community_centers_rel"] = torch.stack(
            community_centers_rel
        )
        self.msg_sender.mailbox["community_scale_rel"] = torch.Tensor(
            [len(p) for p in partition_rel]
        )
        self.msg_sender.mailbox["cur_community_type"] = "rel"
        self.msg_sender.mailbox["cur_etype"] = "create_post_r"
        self._graph.graph.send_and_recv(
            eids_create_post_r_within_window,
            self.msg_sender.msg_community_contribution,
            self.msg_sender.reduce_community_contribution,
            etype=("post", "create_post_r", "user"),
        )
        # Community Type 2: Interaction Based
        history_begin_day = self._graph._history_window_beginning
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
        comment_on_links = self._graph.graph.edges(etype="comment_on")
        comment_on_links_in_window_mask = (
            self._graph.graph.edges["comment_on"].data["tag"] >= history_begin_day
        ) & (self._graph.graph.edges["comment_on"].data["tag"] < history_end_day)
        comment_on_df = pd.DataFrame(
            {
                "user": comment_on_links[0][comment_on_links_in_window_mask].numpy(),
                "post": comment_on_links[1][comment_on_links_in_window_mask].numpy(),
            }
        )
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
        comment_augment_links = comment_on_df.join(
            create_post_df.set_index("post"),
            on="post",
            how="inner",
            lsuffix="_src",
            rsuffix="_dst",
        )
        augment_links = pd.concat(
            [
                like_augment_links[["user_src", "user_dst"]],
                repost_augment_links[["user_src", "user_dst"]],
                comment_augment_links[["user_src", "user_dst"]],
            ]
        ).values
        augment_links = [[int(a[0]), int(a[1])] for a in augment_links if a[0] != a[1]]
        augment_links = np.unique(np.array(augment_links), axis=0)
        sub_g_inter = dgl.graph((augment_links[:, 0], augment_links[:, 1]))
        nx_g_inter = dgl.to_networkx(sub_g_inter)
        partition_inter = nx.community.louvain_communities(nx_g_inter)
        self._graph.graph.nodes["user"].data["partition_inter"] = torch.zeros(
            self._graph.graph.num_nodes("user")
        ).int()
        for i, p in enumerate(partition_inter):
            self._graph.graph.nodes["user"].data["partition_inter"][list(p)] = i

        community_centers_inter = []
        for p in partition_inter:
            community_centers_inter.append(
                self._graph.graph.nodes["user"].data["embedding"][torch.Tensor(list(p)).long()].mean(0)
            )
        self.msg_sender.mailbox["community_centers_inter"] = torch.stack(
            community_centers_inter
        )
        self.msg_sender.mailbox["community_scale_inter"] = torch.Tensor(
            [len(p) for p in partition_inter]
        )
        self.msg_sender.mailbox["cur_community_type"] = "inter"
        self.msg_sender.mailbox["cur_etype"] = "create_post_r"
        self._graph.graph.send_and_recv(
            eids_create_post_r_within_window,
            self.msg_sender.msg_community_contribution,
            self.msg_sender.reduce_community_contribution,
            etype=("post", "create_post_r", "user"),
        )

        # Self-status Seeking - 1 Impress others - Avg & Max Self Post Popularity
        self._logger.debug("Collecting Self-status Seeking - 1 Impress others")
        # user <-[create_post_r]- post
        self.msg_sender.mailbox["cur_rel"] = "post"
        self.msg_sender.mailbox["cur_etype"] = "create_post_r"
        self._graph.graph.send_and_recv(
            eids_create_post_r_within_window,
            self.msg_sender.msg_interaction_popularity,
            self.msg_sender.reduce_interaction_popularity,
            etype=("post", "create_post_r", "user"),
        )

        # Self-status Seeking - 2 Feel Important - Number of Followers
        self._logger.debug("Collecting Self-status Seeking - 2 Feel Important")
        self._graph.graph.nodes["user"].data[
            "num_followers"
        ] = self._graph.graph.in_degrees(etype="follow").float()

        # Self-status Seeking - 3 Make Self Cool & Trendy - Avg & Max Similarity between Posts and Top-K Popular Posts
        self._logger.debug("Collecting Self-status Seeking - 3 Make Self Cool & Trendy")
        top_k_pop_posts = torch.topk(
            self._graph.graph.nodes["post"].data["popularity"], k=self._top_k_pop
        )
        self.msg_sender.mailbox["top_k_pop_posts_emb"] = self._graph.graph.nodes["post"].data["embedding"][top_k_pop_posts.indices]
        self.msg_sender.mailbox["cur_etype"] = "create_post_r"
        self._graph.graph.send_and_recv(
            eids_create_post_r_within_window,
            self.msg_sender.msg_trendy_cossim,
            self.msg_sender.reduce_trendy_cossim,
            etype=("post", "create_post_r", "user"),
        )

        # Social Interaction - 1 Stay in Touch - Number of Friends (mutual-follow)
        self._logger.debug("Collecting Social Interaction - 1 Stay in Touch")
        A = self._graph.graph.adj(etype="follow")
        A = torch.sparse.FloatTensor(A.indices().long(), A.val, A.shape).to_dense()
        self._graph.graph.nodes["user"].data["num_friends"] = (A @ A).sum(1).float()
        del A  # release cache

        # Social Interaction - 2 Meeting Interesting People - Avg & Max Cos Similarity of Following
        self._logger.debug(
            "Collecting Social Interaction - 2 Meeting Interesting People"
        )
        self._graph.graph.update_all(
            self.msg_sender.msg_following_similarity,
            self.msg_sender.reduce_following_similarity,
            etype=("user", "follow_r", "user"),
        )

        # Social Interaction - 3 Value Sharing - Avg Cos Similarity of My Followers's Post - My Posts
        self._logger.debug("Collecting Social Interaction - 3 Value Sharing")
        self._graph.graph.update_all(
            self.msg_sender.msg_value_sharing,
            self.msg_sender.reduce_value_sharing,
            etype=("user", "follow_r", "user"),
        )
        del self._graph.graph.nodes["user"].data[
            "tmpvar_avg_post_embedding"
        ]  # release cache

        # Social Interaction - 4 Belong to Community - Cos Similarity of CommunityCenter-MyInterest and Community Scale
        self._logger.debug("Collecting Social Interaction - 4 Belong to Community")
        self.msg_sender.mailbox["cur_community_type"] = "rel"
        self._graph.graph.apply_nodes(
            self.msg_sender.reduce_community_belonging,
            ntype="user",
        )
        self.msg_sender.mailbox["cur_community_type"] = "inter"
        self._graph.graph.apply_nodes(
            self.msg_sender.reduce_community_belonging,
            ntype="user",
        )
        if "ugt_state" in self._graph.graph.nodes["user"].data:
            self._graph.graph.nodes["user"].data["last_ugt_state"] = self._graph.graph.nodes["user"].data["ugt_state"]
        self._graph.graph.nodes["user"].data["ugt_state"] = torch.concat(
            [
                # === Use Features ===
                self._graph.graph.nodes["user"].data["post_count"],  # feature-0~6
                self._graph.graph.nodes["user"].data["repost_count"],  # feature-7~13
                self._graph.graph.nodes["user"].data["comment_count"],  # feature-14~20
                self._graph.graph.nodes["user"].data["like_count"],  # feature-21~27
                # === Gratification Features ===
                # Seeking Info - 1 Something Interesting
                self._graph.graph.nodes["user"].data["avg_intered_similarity_like"].unsqueeze(-1),  # feature-28
                self._graph.graph.nodes["user"].data["max_intered_similarity_like"].unsqueeze(-1),  # feature-29
                self._graph.graph.nodes["user"].data["avg_intered_similarity_repost"].unsqueeze(-1),  # feature-30
                self._graph.graph.nodes["user"].data["max_intered_similarity_repost"].unsqueeze(-1),  # feature-31
                self._graph.graph.nodes["user"].data["avg_intered_similarity_comment"].unsqueeze(-1),  # feature-32
                self._graph.graph.nodes["user"].data["max_intered_similarity_comment"].unsqueeze(-1),  # feature-33
                # Seeking Info - 2 Something Useful
                2 * self._graph.graph.nodes["user"].data["avg_intered_popularity_like"].unsqueeze(-1).sigmoid() - 1,  # feature-34
                2 * self._graph.graph.nodes["user"].data["max_intered_popularity_like"].unsqueeze(-1).sigmoid() - 1,  # feature-35
                2 * self._graph.graph.nodes["user"].data["avg_intered_popularity_repost"].unsqueeze(-1).sigmoid() - 1,  # feature-36
                2 * self._graph.graph.nodes["user"].data["max_intered_popularity_repost"].unsqueeze(-1).sigmoid() - 1,  # feature-37
                2 * self._graph.graph.nodes["user"].data["avg_intered_popularity_comment"].unsqueeze(-1).sigmoid() - 1,  # feature-38
                2 * self._graph.graph.nodes["user"].data["max_intered_popularity_comment"].unsqueeze(-1).sigmoid() - 1,  # feature-39
                # Seeking Info - 3 Something New
                (self._graph.graph.nodes["user"].data["avg_post_time_interval"].unsqueeze(-1) - 1)/6,  # feature-40
                (self._graph.graph.nodes["user"].data["avg_repost_time_interval"].unsqueeze(-1) - 1)/6,  # feature-41
                (self._graph.graph.nodes["user"].data["avg_comment_time_interval"].unsqueeze(-1) - 1)/6,  # feature-42
                (self._graph.graph.nodes["user"].data["avg_like_time_interval"].unsqueeze(-1) - 1)/6,  # feature-43
                # Sharing Info - 1 Self Expression
                self._graph.graph.nodes["user"].data["avg_published_similarity_post"].unsqueeze(-1),  # feature-44
                self._graph.graph.nodes["user"].data["max_published_similarity_post"].unsqueeze(-1),  # feature-45
                self._graph.graph.nodes["user"].data["avg_published_similarity_repost"].unsqueeze(-1),  # feature-46
                self._graph.graph.nodes["user"].data["max_published_similarity_repost"].unsqueeze(-1),  # feature-47
                self._graph.graph.nodes["user"].data["avg_published_similarity_comment"].unsqueeze(-1),  # feature-48
                self._graph.graph.nodes["user"].data["max_published_similarity_comment"].unsqueeze(-1),  # feature-49
                # Sharing Info - 2 Provide to Others
                self._graph.graph.nodes["user"].data["avg_cross_follower_and_post_similarity"].unsqueeze(-1),  # feature-50
                # Sharing Info - 3 Contribute to Community
                self._graph.graph.nodes["user"].data["avg_community_contribution_rel"].unsqueeze(-1),  # feature-51
                self._graph.graph.nodes["user"].data["max_community_contribution_rel"].unsqueeze(-1),  # feature-52
                self._graph.graph.nodes["user"].data["avg_community_contribution_inter"].unsqueeze(-1),  # feature-53
                self._graph.graph.nodes["user"].data["max_community_contribution_inter"].unsqueeze(-1),  # feature-54
                # Self-status Seeking - 1 Impress others
                2 * self._graph.graph.nodes["user"].data["avg_published_popularity_post"].unsqueeze(-1).sigmoid() - 1,  # feature-55
                2 * self._graph.graph.nodes["user"].data["max_published_popularity_post"].unsqueeze(-1).sigmoid() - 1,  # feature-56
                # Self-status Seeking - 2 Feel Important
                2 * self._graph.graph.nodes["user"].data["num_followers"].unsqueeze(-1).sigmoid() - 1,  # feature-57
                # Self-status Seeking - 3 Make Self Cool & Trendy
                self._graph.graph.nodes["user"].data["avg_trend_similarity"].unsqueeze(-1),  # feature-58
                self._graph.graph.nodes["user"].data["max_trend_similarity"].unsqueeze(-1),  # feature-59
                # Social Interaction - 1 Stay in Touch
                2 * self._graph.graph.nodes["user"].data["num_friends"].unsqueeze(-1).sigmoid() - 1,  # feature-60
                # Social Interaction - 2 Meeting Interesting People
                self._graph.graph.nodes["user"].data["avg_following_similarity"].unsqueeze(-1),  # feature-61
                self._graph.graph.nodes["user"].data["max_following_similarity"].unsqueeze(-1),  # feature-62
                # Social Interaction - 3 Value Sharing
                self._graph.graph.nodes["user"].data["avg_post_value_sharing"].unsqueeze(-1),  # feature-63
                # Social Interaction - 4 Belong to Community
                self._graph.graph.nodes["user"].data["community_belonging_rel"].unsqueeze(-1),  # feature-64
                2 * self._graph.graph.nodes["user"].data["community_scale_rel"].unsqueeze(-1).sigmoid() - 1,  # feature-65
                self._graph.graph.nodes["user"].data["community_belonging_inter"].unsqueeze(-1),  # feature-66
                2 * self._graph.graph.nodes["user"].data["community_scale_inter"].unsqueeze(-1).sigmoid() - 1,  # feature-67
            ],
            dim=1,
        )
        self._metrics_service.append("ugt_state", self._graph.graph.nodes["user"].data["ugt_state"])

    # Trained Model Required
    def update_action_density_and_threshold(self):
        ugt_state = self._graph.graph.nodes["user"].data["ugt_state"]
        last_ugt_state = self._graph.graph.nodes["user"].data["last_ugt_state"]

        for action_type in ["post", "repost", "comment", "like"]:
            mutual_info_value = self._graph.graph.nodes["user"].data[f"{action_type}_mi"]
            self._graph.graph.nodes["user"].data[f"{action_type}_density_this_round"] = self._user_action_service.pred_action_density(
                action_type, ugt_state, last_ugt_state, mutual_info_value
            )
            self._metrics_service.append(f"{action_type}_density_this_round", self._graph.graph.nodes["user"].data[f"{action_type}_density_this_round"])
        for action_type in ["repost", "comment", "like"]:
            self._graph.graph.nodes["user"].data[f"{action_type}_threshold"] = self._user_action_service.density_2_threshold(
                action_type, self._graph.graph.nodes["user"].data[f"{action_type}_density_this_round"]
            )
            self._metrics_service.append(f"{action_type}_threshold", self._graph.graph.nodes["user"].data[f"{action_type}_threshold"])

    # Trained Model Required
    def browse(self, posts_recommendation):
        self._logger.debug("User Browsing ...")
        user_browse_actions = {}
        post_embedding = self._graph.graph.nodes["post"].data["embedding"]
        user_history_embedding = self._graph.graph.nodes["user"].data['history_interacted']
        user_embedding = self._graph.graph.nodes["user"].data['embedding']
        num_user = user_history_embedding.shape[0]
        user_topic_engage = self._graph.graph.nodes["user"].data["topic_engage_post"] + self._graph.graph.nodes["user"].data["topic_engage_repost"] + self._graph.graph.nodes["user"].data["topic_engage_comment"] + self._graph.graph.nodes["user"].data["topic_engage_like"]

        # === POST ===
        post_density_this_round = self._graph.graph.nodes["user"].data["post_density_this_round"]
        post_this_round = []
        self._logger.debug("User Posting ...")
        for uid in tqdm(range(num_user), disable=not self._config.show_progress):
            user_post_num = min(int(post_density_this_round[uid]), self._max_post_num_per_round)
            for j in range(user_post_num):
                post_this_round.append(uid)
        user_browse_actions['new_post_relations'] = post_this_round  # uid only
        posting_user_ids = torch.tensor(post_this_round).long()
        new_post_topic = self._user_action_service.pred_post_topic_ID(user_topic_engage[posting_user_ids])
        user_browse_actions['new_post_topic_id'] = new_post_topic
        new_post_emb = self._user_action_service.generate(new_post_topic)
        user_browse_actions['new_post_embedding'] = new_post_emb
        self._logger.debug("{} Posts are Posted ...".format(len(post_this_round)))

        # === LIKE ===
        self._logger.debug("User Giving Likes ...")
        # Rank
        like_scores = []
        for i in tqdm(range(int(len(posts_recommendation)/self._simu_batch_size)+1), disable=not self._config.show_progress):
            begin = i * self._simu_batch_size
            end = (i + 1) * self._simu_batch_size
            batch = (
                user_history_embedding[posts_recommendation[begin:end, 0]].to(self._config.device), 
                user_embedding[posts_recommendation[begin:end, 0]].to(self._config.device),
                post_embedding[posts_recommendation[begin:end, 1]].to(self._config.device), 
            )
            like_score = self._user_action_service.score("like", batch).detach().cpu()
            like_scores.append(like_score)
        like_scores = torch.cat(like_scores, dim=0)
        # Interact with beyond threshold
        user_like_threshold = self._graph.graph.nodes["user"].data[f"like_threshold"][posts_recommendation[:, 0]]
        do_like_musk = like_scores > user_like_threshold
        like_this_round = [[int(uid), int(nid)] for uid, nid in posts_recommendation[do_like_musk]]
        user_browse_actions['new_like_relations'] = like_this_round
        # Response - Pass for like
        # Add Corresponding Authors into Candidate Friend List
        liked_posts_ids = torch.tensor([nid for uid, nid in like_this_round]).long()
        author_of_liked_news = self._graph.graph.adj('create_post_r').indices()[1][liked_posts_ids]
        user_browse_actions['candidate_friend_like'] = [[int(uid), int(aid)] for uid, aid in zip(torch.tensor(like_this_round)[:, 0], author_of_liked_news)]
        self._logger.debug("{} Likes are Given ...".format(len(like_this_round)))
        
        # === COMMENT ===
        self._logger.debug("User Giving Comments ...")
        # Rank
        comment_scores = []
        for i in tqdm(range(int(len(posts_recommendation)/self._simu_batch_size)+1), disable=not self._config.show_progress):
            begin = i * self._simu_batch_size
            end = (i + 1) * self._simu_batch_size
            batch = (
                user_history_embedding[posts_recommendation[begin:end, 0]].to(self._config.device), 
                user_embedding[posts_recommendation[begin:end, 0]].to(self._config.device),
                post_embedding[posts_recommendation[begin:end, 1]].to(self._config.device), 
            )
            comment_score = self._user_action_service.score("comment", batch).detach().cpu()
            comment_scores.append(comment_score)
        comment_scores = torch.cat(comment_scores, dim=0)
        # Interact with beyond threshold
        user_comment_threshold = self._graph.graph.nodes["user"].data[f"comment_threshold"][posts_recommendation[:, 0]]
        do_comment_musk = comment_scores > user_comment_threshold
        comment_this_round = [[int(uid), int(nid)] for uid, nid in posts_recommendation[do_comment_musk]]
        user_browse_actions['new_comment_relations'] = comment_this_round
        # Response
        commented_posts_ids = torch.tensor([nid for uid, nid in comment_this_round]).long()
        commenting_user_ids = torch.tensor([uid for uid, nid in comment_this_round]).long()
        commented_posts_topic_id = self._graph.graph.nodes["post"].data["topic_id"][commented_posts_ids]
        commenting_user_topic_engage = self._graph.graph.nodes["user"].data["topic_engage_comment"][commenting_user_ids]
        user_browse_actions['new_comment_topic_id'] = commented_posts_topic_id
        new_comment_emb = self._user_action_service.generate(commented_posts_topic_id)
        user_browse_actions['new_comment_embedding'] = new_comment_emb
        # Add Corresponding Authors into Candidate Friend List
        author_of_commented_news = self._graph.graph.adj('create_post_r').indices()[1][commented_posts_ids]
        user_browse_actions['candidate_friend_comment'] = [[int(uid), int(aid)] for uid, aid in zip(torch.tensor(comment_this_round)[:, 0], author_of_commented_news)]
        self._logger.debug("{} Comments are Given ...".format(len(comment_this_round)))

        # === REPOST ===
        self._logger.debug("User Reposting ...")
        # Rank
        repost_scores = []
        for i in tqdm(range(int(len(posts_recommendation)/self._simu_batch_size)+1), disable=not self._config.show_progress):
            begin = i * self._simu_batch_size
            end = (i + 1) * self._simu_batch_size
            batch = (
                user_history_embedding[posts_recommendation[begin:end, 0]].to(self._config.device), 
                user_embedding[posts_recommendation[begin:end, 0]].to(self._config.device),
                post_embedding[posts_recommendation[begin:end, 1]].to(self._config.device), 
            )
            repost_score = self._user_action_service.score("repost", batch).detach().cpu()
            repost_scores.append(repost_score)
        repost_scores = torch.cat(repost_scores, dim=0)
        # Interact with beyond threshold
        user_repost_threshold = self._graph.graph.nodes["user"].data[f"repost_threshold"][posts_recommendation[:, 0]]
        do_repost_musk = repost_scores > user_repost_threshold
        repost_this_round = [[int(uid), int(nid)] for uid, nid in posts_recommendation[do_repost_musk]]
        user_browse_actions['new_repost_relations'] = repost_this_round
        # Response
        reposted_posts_ids = torch.tensor([nid for uid, nid in repost_this_round]).long()
        reposting_user_ids = torch.tensor([uid for uid, nid in repost_this_round]).long()
        reposted_posts_topic_id = self._graph.graph.nodes["post"].data["topic_id"][reposted_posts_ids]
        reposting_user_topic_engage = self._graph.graph.nodes["user"].data["topic_engage_repost"][reposting_user_ids]
        user_browse_actions['new_repost_topic_id'] = reposted_posts_topic_id
        new_repost_emb = self._user_action_service.generate(reposted_posts_topic_id)
        user_browse_actions['new_repost_embedding'] = new_repost_emb
        # Add Corresponding Authors into Candidate Friend List
        author_of_reposted_news = self._graph.graph.adj('create_post_r').indices()[1][reposted_posts_ids]
        user_browse_actions['candidate_friend_repost'] = [[int(uid), int(aid)] for uid, aid in zip(torch.tensor(repost_this_round)[:, 0], author_of_reposted_news)]
        self._logger.debug("{} Reposts are Given ...".format(len(repost_this_round)))

        return user_browse_actions

    # Trained Model Required
    def follow(self, friends_recommendation, user_browse_actions):
        self._logger.debug("User Following ...")
        user_follow_actions = {}  # new_follow_relations
        new_follow_relations = []
        user_history_embedding = self._graph.graph.nodes["user"].data['history_interacted']
        user_embedding = self._graph.graph.nodes["user"].data['embedding']
        num_user = user_history_embedding.shape[0]
        length_history_seq = user_history_embedding.shape[1]
        
        # From Recommendation
        candidate_friends_from_recommendation = friends_recommendation.long()
        num_candidate_pairs = candidate_friends_from_recommendation.shape[0]
        follow_scores = []
        for i in tqdm(range(int(num_candidate_pairs/self._simu_batch_size)+1), disable=not self._config.show_progress):
            begin = i * self._simu_batch_size
            end = (i + 1) * self._simu_batch_size
            user_index = candidate_friends_from_recommendation[begin:end][:, 0]
            candidate_index = candidate_friends_from_recommendation[begin:end][:, 1]
            user_history_embedding = self._graph.graph.nodes["user"].data['history_interacted'][user_index]
            candidate_history_embedding = self._graph.graph.nodes["user"].data['history_interacted'][candidate_index]
            user_embedding = self._graph.graph.nodes["user"].data['embedding'][user_index]
            candidate_embedding = self._graph.graph.nodes["user"].data['embedding'][candidate_index]
            batch = (
                user_history_embedding.to(self._config.device), 
                user_embedding.to(self._config.device), 
                candidate_history_embedding.to(self._config.device), 
                candidate_embedding.to(self._config.device),
            )
            follow_score = self._user_action_service.score("follow", batch).detach().cpu()
            follow_scores.append(follow_score)
        follow_scores = torch.cat(follow_scores, dim=0)
        make_friends_musk = follow_scores > self._graph.graph.nodes["user"].data["follow_threshold"][candidate_friends_from_recommendation[:, 0].long()]
        new_follow_relations.extend([[int(candidate_friends_from_recommendation[i][0]), int(candidate_friends_from_recommendation[i][1])] for i in range(num_candidate_pairs) if make_friends_musk[i]])
        
        # From Interaction
        candidate_friends_from_interaction = user_browse_actions['candidate_friend_like'] + user_browse_actions['candidate_friend_comment'] + user_browse_actions['candidate_friend_repost']
        candidate_friends_from_interaction = torch.tensor(candidate_friends_from_interaction).long()
        num_candidate_pairs = candidate_friends_from_interaction.shape[0]
        follow_scores = []
        for i in tqdm(range(int(num_candidate_pairs/self._simu_batch_size)+1), disable=not self._config.show_progress):
            begin = i * self._simu_batch_size
            end = (i + 1) * self._simu_batch_size
            user_index = candidate_friends_from_interaction[begin:end][:, 0]
            candidate_index = candidate_friends_from_interaction[begin:end][:, 1]
            user_history_embedding = self._graph.graph.nodes["user"].data['history_interacted'][user_index]
            candidate_history_embedding = self._graph.graph.nodes["user"].data['history_interacted'][candidate_index]
            user_embedding = self._graph.graph.nodes["user"].data['embedding'][user_index]
            candidate_embedding = self._graph.graph.nodes["user"].data['embedding'][candidate_index]
            batch = (
                user_history_embedding.to(self._config.device), 
                user_embedding.to(self._config.device), 
                candidate_history_embedding.to(self._config.device), 
                candidate_embedding.to(self._config.device),
            )
            follow_score = self._user_action_service.score("follow", batch).detach().cpu()
            follow_scores.append(follow_score)
        follow_scores = torch.cat(follow_scores, dim=0)
        make_friends_musk = follow_scores > self._graph.graph.nodes["user"].data["follow_threshold"][candidate_friends_from_interaction[:, 0].long()]
        new_follow_relations.extend([[int(candidate_friends_from_interaction[i][0]), int(candidate_friends_from_interaction[i][1])] for i in range(num_candidate_pairs) if make_friends_musk[i]])

        user_follow_actions['new_follow_relations'] = new_follow_relations
        self._logger.debug("{} Follows are Given ...".format(len(new_follow_relations)))

        self._logger.debug("User Unfollowing ...")
        batch_size = self._simu_batch_size
        following_relations = self._graph.graph.adj('follow').indices()
        src_user_history_embedding = self._graph.graph.nodes["user"].data['history_interacted'][following_relations[0]]
        dst_user_history_embedding = self._graph.graph.nodes["user"].data['history_interacted'][following_relations[1]]
        src_user_embedding = self._graph.graph.nodes["user"].data['embedding'][following_relations[0]]
        dst_user_embedding = self._graph.graph.nodes["user"].data['embedding'][following_relations[1]]
        num_batch = int(following_relations.shape[1]/batch_size) + 1
        follow_scores = []
        for i in tqdm(range(num_batch), disable=not self._config.show_progress):
            begin = i * batch_size
            end = (i + 1) * batch_size
            batch = (
                src_user_history_embedding[begin:end].to(self._config.device), 
                src_user_embedding[begin:end].to(self._config.device), 
                dst_user_history_embedding[begin:end].to(self._config.device), 
                dst_user_embedding[begin:end].to(self._config.device),
            )
            follow_score = self._user_action_service.score("follow", batch).detach().cpu()
            follow_scores.append(follow_score)
        follow_scores = torch.cat(follow_scores, dim=0)
        unfollow_musk = follow_scores < self._graph.graph.nodes["user"].data["unfollow_threshold"][following_relations[0].long()]
        unfollow_relations = following_relations[:, unfollow_musk].T.numpy().tolist()
        user_follow_actions['unfollow_relations'] = unfollow_relations
        self._logger.debug("{} Follows are Unfollowed ...".format(len(unfollow_relations)))

        return user_follow_actions


class MsgSender:
    def __init__(self, graph):
        self._graph = graph
        self.mailbox = {}

    def edges_with_tag(self, edges):
        tag = self.mailbox["tag"]
        return edges.data['tag'] == tag
    
    def edges_within_window(self, edges):
        window_begin = self.mailbox["window_begin"]
        window_end = self.mailbox["window_end"]
        edge_tag = edges.data['tag']
        return (edge_tag >= window_begin) & (edge_tag < window_end)
    
    def msg_sample_user_history_interactions_by_time(self, edges):
        return {
            "edge_tag": edges.data["tag"],
            'tmpvar_embedding': edges.src['embedding'],
            "tmpvar_topic_id": edges.src["topic_id"],
            'tmpvar_ID': edges.src['_ID'],
        }

    def reduce_sample_user_history_interactions_by_time(self, nodes):
        edge_tag = nodes.mailbox["edge_tag"].long()
        tmpvar_embedding = nodes.mailbox['tmpvar_embedding']
        tmpvar_topic_id = nodes.mailbox['tmpvar_topic_id']
        tmpvar_ID = nodes.mailbox['tmpvar_ID']
        
        time_interval = self.mailbox["window_end"] - edge_tag
        sample_weight = torch.exp(-time_interval.float())
        sampled_index = torch.Tensor(
            [list(torch.utils.data.WeightedRandomSampler(weight, self._history_length, replacement=True)) for weight in sample_weight]
        ).long()
        sampled_ID = torch.stack([tmpvar_ID[i][sampled_index[i]] for i in range(sampled_index.shape[0])], dim=0)
        sampled_index_one_hot = torch.nn.functional.one_hot(sampled_index, num_classes=tmpvar_embedding.shape[1])
        
        gmm_signal = torch.zeros(tmpvar_topic_id.shape[0], self._gmm_topic_num)
        for i, user_gmm_topic in enumerate(tmpvar_topic_id):
            for user_gmm_topic_index in user_gmm_topic:
                gmm_signal[i, user_gmm_topic_index] += 1
        
        return {
            'history_{}'.format(self.mailbox["cur_processing_action"]): torch.matmul(sampled_index_one_hot.float(), tmpvar_embedding),
            'history_{}_ID'.format(self.mailbox["cur_processing_action"]): sampled_ID,
            'topic_engage_{}'.format(self.mailbox["cur_processing_action"]): gmm_signal,
        }

    def msg_sample_user_history_interactions_by_through_pass(self, edges):
        return {
            "tmpvar_through_pass": edges.data["through_pass"],
            'tmpvar_embedding': edges.src['embedding'],
            "tmpvar_topic_id": edges.src["topic_id"],
            'tmpvar_ID': edges.src['_ID'],
        }

    def reduce_sample_user_history_interactions_by_through_pass(self, nodes):
        tmpvar_through_pass = nodes.mailbox["tmpvar_through_pass"]
        tmpvar_embedding = nodes.mailbox['tmpvar_embedding']
        tmpvar_topic_id = nodes.mailbox['tmpvar_topic_id']
        tmpvar_ID = nodes.mailbox['tmpvar_ID']
        
        sampled_index = torch.Tensor(
            [list(torch.utils.data.WeightedRandomSampler(weight.sigmoid(), self._history_length, replacement=True)) for weight in tmpvar_through_pass]
        ).long()
        sampled_ID = torch.stack([tmpvar_ID[i][sampled_index[i]] for i in range(sampled_index.shape[0])], dim=0)
        sampled_index_one_hot = torch.nn.functional.one_hot(sampled_index, num_classes=tmpvar_embedding.shape[1])
        
        gmm_signal = torch.zeros(tmpvar_topic_id.shape[0], self._gmm_topic_num)
        for i, user_gmm_topic in enumerate(tmpvar_topic_id):
            for user_gmm_topic_index in user_gmm_topic:
                gmm_signal[i, user_gmm_topic_index] += 1
        
        return {
            'history_{}'.format(self.mailbox["cur_processing_action"]): torch.matmul(sampled_index_one_hot.float(), tmpvar_embedding),
            'history_{}_ID'.format(self.mailbox["cur_processing_action"]): sampled_ID,
            'topic_engage_{}'.format(self.mailbox["cur_processing_action"]): gmm_signal,
        }

    def msg_count_usage(self, edges):
        return {
            "edge_tag": edges.data["tag"] - self.mailbox["window_begin"], 
        }

    def reduce_count_usage(self, nodes):
        return {
            "count_usage": torch.nn.functional.one_hot(nodes.mailbox["edge_tag"].long(), num_classes=self._window_size).sum(dim=1),
        }

    def msg_interaction_cossim_and_popularity(self, edges):
        # Avg & Max Interacted Post Similarity
        # user <-[like_r]- post
        return {
            "edge_tag": edges.data["tag"],
            "tmpvar_embedding": edges.src["embedding"],
            "tmpvar_popularity": edges.src["popularity"],
            'tmpvar_ID': edges.data['_ID'],
        }

    def reduce_interaction_cossim_and_popularity(self, nodes):
        # Avg & Max Interacted Post Similarity
        edge_tag = nodes.mailbox["edge_tag"].long()
        tmpvar_embedding = nodes.mailbox["tmpvar_embedding"]
        tmpvar_popularity = nodes.mailbox["tmpvar_popularity"]
        tmpvar_ID = nodes.mailbox["tmpvar_ID"]
        cur_rel = self.mailbox["cur_rel"]
        cur_etype = self.mailbox["cur_etype"]

        user_embedding = nodes.data["embedding"].unsqueeze(1)
        user_embedding_cosine_similarity = F.cosine_similarity(
            user_embedding, tmpvar_embedding, dim=-1
        )
        avg_embedding_cosine_similarity = user_embedding_cosine_similarity.mean(-1)
        max_embedding_cosine_similarity = user_embedding_cosine_similarity.max(-1)[0]

        popularity_in_window = tmpvar_popularity
        avg_popularity = popularity_in_window.mean(-1)
        max_popularity = popularity_in_window.max(-1)[0]

        self._graph.graph.edges[cur_etype].data['through_pass'][tmpvar_ID.reshape(-1).long()] += user_embedding_cosine_similarity.reshape(-1)
        self._graph.graph.edges[cur_etype].data['through_pass'][tmpvar_ID.reshape(-1).long()] += 2 * tmpvar_popularity.reshape(-1).sigmoid() - 1

        return {
            "avg_intered_similarity_{}".format(cur_rel): avg_embedding_cosine_similarity,
            "max_intered_similarity_{}".format(cur_rel): max_embedding_cosine_similarity,
            "avg_intered_popularity_{}".format(cur_rel): avg_popularity,
            "max_intered_popularity_{}".format(cur_rel): max_popularity,
        }

    def msg_repeat_emb_and_popularity(self, edges):
        # Avg & Max Interacted Post Similarity
        # Embedding 消息中继
        # user <-[create_repost_r]- repost <-[repost_of_r]- post
        # user <-[create_comment_r]- comment <-[comment_of_r]- post
        return {
            "tmpvar_embedding": edges.src["embedding"],
            "tmpvar_popularity": edges.src["popularity"],
        }

    def reduce_repeat_emb_and_popularity(self, nodes):
        # Avg & Max Interacted Post Similarity
        # Embedding 消息中继
        # user <-[create_repost_r]- repost <-[repost_of_r]- post
        # user <-[create_comment_r]- comment <-[comment_of_r]- post
        return {
            "repeat_embedding": nodes.mailbox["tmpvar_embedding"].mean(1),
            "repeat_popularity": nodes.mailbox["tmpvar_popularity"].mean(1),
        }

    def msg_interaction_cossim_and_popularity_from_repeat(self, edges):
        # Avg & Max Interacted Post Similarity
        # Embedding via 消息中继
        # user <-[create_repost_r]- repost <-[repost_of_r]- post
        # user <-[create_comment_r]- comment <-[comment_of_r]- post
        return {
            "edge_tag": edges.data["tag"],
            "tmpvar_embedding": edges.src["repeat_embedding"],
            "tmpvar_popularity": edges.src["repeat_popularity"],
            'tmpvar_ID': edges.data['_ID'],
        }

    def msg_interaction_cossim(self, edges):
        # Avg & Max Published Post Similarity
        # user <-[like_r]- post
        return {
            "edge_tag": edges.data["tag"],
            "tmpvar_embedding": edges.src["embedding"],
            'tmpvar_ID': edges.data['_ID'],
        }

    def reduce_interaction_cossim(self, nodes):
        # Avg & Max Published Post Similarity
        edge_tag = nodes.mailbox["edge_tag"].long()
        tmpvar_embedding = nodes.mailbox["tmpvar_embedding"]
        tmpvar_ID = nodes.mailbox["tmpvar_ID"]
        cur_rel = self.mailbox["cur_rel"]
        cur_etype = self.mailbox["cur_etype"]

        user_embedding = nodes.data["embedding"].unsqueeze(1)
        user_embedding_cosine_similarity = F.cosine_similarity(
            user_embedding, tmpvar_embedding, dim=-1
        )
        avg_embedding_cosine_similarity = user_embedding_cosine_similarity.mean(-1)
        max_embedding_cosine_similarity = user_embedding_cosine_similarity.max(-1)[0]

        self._graph.graph.edges[cur_etype].data['through_pass'][tmpvar_ID.reshape(-1).long()] += user_embedding_cosine_similarity.reshape(-1)

        return {
            "avg_published_similarity_{}".format(cur_rel): avg_embedding_cosine_similarity,
            "max_published_similarity_{}".format(cur_rel): max_embedding_cosine_similarity,
        }

    def msg_interaction_popularity(self, edges):
        # Avg & Max Published Post Popularity
        # user <-[like_r]- post
        return {
            "edge_tag": edges.data["tag"],
            "tmpvar_popularity": edges.src["popularity"],
            'tmpvar_ID': edges.data['_ID'],
        }

    def reduce_interaction_popularity(self, nodes):
        # Avg & Max Published Post Popularity
        edge_tag = nodes.mailbox["edge_tag"].long()
        tmpvar_popularity = nodes.mailbox["tmpvar_popularity"]
        tmpvar_ID = nodes.mailbox["tmpvar_ID"]
        cur_rel = self.mailbox["cur_rel"]
        cur_etype = self.mailbox["cur_etype"]

        avg_popularity = tmpvar_popularity.mean(-1)
        max_popularity = tmpvar_popularity.max(-1)[0]

        self._graph.graph.edges[cur_etype].data['through_pass'][tmpvar_ID.reshape(-1).long()] += 2 * tmpvar_popularity.reshape(-1).sigmoid() - 1

        return {
            "avg_published_popularity_{}".format(cur_rel): avg_popularity,
            "max_published_popularity_{}".format(cur_rel): max_popularity,
        }

    def msg_following_similarity(self, edges):
        # Avg & Max Cos Similarity of Following
        # user <-[follow_r]- user
        return {
            "tmpvar_history_avg": edges.src["embedding"],
        }

    def reduce_following_similarity(self, nodes):
        # Avg & Max Cos Similarity of Following
        # user <-[follow_r]- user
        tmpvar_history_avg = nodes.mailbox["tmpvar_history_avg"]
        user_embedding = nodes.data["embedding"].unsqueeze(1)
        embedding_cosine_similarity = F.cosine_similarity(
            user_embedding, tmpvar_history_avg, dim=-1
        )
        avg_embedding_cosine_similarity = embedding_cosine_similarity.mean(-1)
        max_embedding_cosine_similarity = embedding_cosine_similarity.max(-1)[0]
        return {
            "avg_following_similarity": avg_embedding_cosine_similarity,
            "max_following_similarity": max_embedding_cosine_similarity,
        }

    def msg_avg_follower_embedding(self, edges):
        # Avg & Max Cos Similarity of MyFollowers-MyPosts
        # user <-[follow]- user
        # user <-[create_post_r]- post
        return {
            "tmpvar_embedding": edges.src["embedding"],
        }

    def reduce_avg_follower_embedding(self, nodes):
        # Avg & Max Cos Similarity of Following
        # user <-[follow_r]- user
        tmpvar_embedding = nodes.mailbox["tmpvar_embedding"]
        return {
            "tmpvar_avg_follower_embedding": tmpvar_embedding.mean(1),
        }

    def msg_avg_posted_embedding(self, edges):
        # Avg & Max Cos Similarity of MyFollowers-MyPosts
        # user <-[follow]- user
        # user <-[create_post_r]- post
        return {
            "edge_tag": edges.data["tag"],
            "tmpvar_embedding": edges.src["embedding"],
            'tmpvar_ID': edges.data['_ID'],
        }

    def reduce_avg_cross_follower_and_post_similarity(self, nodes):
        # Avg & Max Cos Similarity of Following
        edge_tag = nodes.mailbox["edge_tag"].long()
        tmpvar_embedding = nodes.mailbox["tmpvar_embedding"]
        tmpvar_avg_post_embedding = tmpvar_embedding.mean(1)
        tmpvar_ID = nodes.mailbox["tmpvar_ID"]
        cur_etype = self.mailbox["cur_etype"]
        tmpvar_avg_follower_embedding = nodes.data["tmpvar_avg_follower_embedding"]

        cross_follower_and_post_similarity = F.cosine_similarity(
            tmpvar_avg_follower_embedding.unsqueeze(1),
            tmpvar_embedding,
            dim=-1,
        )

        self._graph.graph.edges[cur_etype].data['through_pass'][tmpvar_ID.reshape(-1).long()] += cross_follower_and_post_similarity.reshape(-1)

        return {
            "tmpvar_avg_post_embedding": tmpvar_avg_post_embedding,
            "avg_cross_follower_and_post_similarity": cross_follower_and_post_similarity.mean(-1)
        }

    def msg_value_sharing(self, edges):
        # My Post ~ My Following's Post Similarity
        return {
            "tmpvar_following_avg_post_embedding": edges.src[
                "tmpvar_avg_post_embedding"
            ],
        }

    def reduce_value_sharing(self, nodes):
        # My Post ~ My Following's Post Similarity
        # Some following may not have post in the window
        tmpvar_following_avg_post_embedding = nodes.mailbox[
            "tmpvar_following_avg_post_embedding"
        ]
        user_avg_post_embedding = nodes.data["tmpvar_avg_post_embedding"].unsqueeze(1)
        not_empty_mask = (tmpvar_following_avg_post_embedding.sum(-1) != 0).int()
        value_sharing = F.cosine_similarity(
            tmpvar_following_avg_post_embedding, user_avg_post_embedding, dim=-1
        )
        avg_post_value_sharing = value_sharing.sum(1) / (not_empty_mask.sum(1) + 1e-8)
        return {
            "avg_post_value_sharing": avg_post_value_sharing,
        }

    def msg_trendy_cossim(self, edges):
        # Avg & Max Published Post Similarity
        # user <-[like_r]- post
        return {
            "edge_tag": edges.data["tag"],
            "tmpvar_embedding": edges.src["embedding"],
            'tmpvar_ID': edges.data['_ID'],
        }

    def reduce_trendy_cossim(self, nodes):
        # Avg & Max Published Post Similarity
        edge_tag = nodes.mailbox["edge_tag"].long()
        tmpvar_embedding = nodes.mailbox["tmpvar_embedding"]
        trendy_post_embedding = self.mailbox["top_k_pop_posts_emb"]
        tmpvar_ID = nodes.mailbox["tmpvar_ID"]
        cur_etype = self.mailbox["cur_etype"]

        neighbor_size = tmpvar_embedding.shape[1]

        embedding_cosine_similarity = F.cosine_similarity(
            trendy_post_embedding.unsqueeze(0),
            tmpvar_embedding.reshape(-1, 1, 512),
            dim=-1,
        ).reshape(-1, neighbor_size * self._top_k_pop)

        avg_embedding_cosine_similarity = embedding_cosine_similarity.mean(-1)
        max_embedding_cosine_similarity = embedding_cosine_similarity.max(-1)[0]

        self._graph.graph.edges[cur_etype].data['through_pass'][tmpvar_ID.reshape(-1).long()] += embedding_cosine_similarity.reshape(-1, self._top_k_pop).mean(-1)

        return {
            "avg_trend_similarity": avg_embedding_cosine_similarity,
            "max_trend_similarity": max_embedding_cosine_similarity,
        }

    def msg_community_contribution(self, edges):
        return {
            "edge_tag": edges.data["tag"],
            "tmpvar_embedding": edges.src["embedding"],
            'tmpvar_ID': edges.data['_ID'],
        }

    def reduce_community_contribution(self, nodes):
        # Avg & Max Published Post Similarity
        edge_tag = nodes.mailbox["edge_tag"].long()
        tmpvar_embedding = nodes.mailbox["tmpvar_embedding"]
        cur_community_type = self.mailbox["cur_community_type"]
        tmpvar_ID = nodes.mailbox["tmpvar_ID"]
        cur_etype = self.mailbox["cur_etype"]

        community_centers = self.mailbox[
            "community_centers_{}".format(cur_community_type)
        ]
        community_centers_per_user = community_centers[
            nodes.data["partition_{}".format(cur_community_type)].long()
        ]
        embedding_cosine_similarity = F.cosine_similarity(
            community_centers_per_user.unsqueeze(1), tmpvar_embedding, dim=-1
        )

        avg_embedding_cosine_similarity = embedding_cosine_similarity.mean(-1)
        max_embedding_cosine_similarity = embedding_cosine_similarity.max(-1)[0]

        self._graph.graph.edges[cur_etype].data['through_pass'][tmpvar_ID.reshape(-1).long()] += embedding_cosine_similarity.reshape(-1)

        return {
            "avg_community_contribution_{}".format(cur_community_type): avg_embedding_cosine_similarity,
            "max_community_contribution_{}".format(cur_community_type): max_embedding_cosine_similarity,
        }

    def reduce_community_belonging(self, nodes):
        cur_community_type = self.mailbox["cur_community_type"]

        user_embedding = nodes.data["embedding"]
        community_centers = self.mailbox["community_centers_{}".format(cur_community_type)]
        community_centers_per_user = community_centers[
            nodes.data["partition_{}".format(cur_community_type)].long()
        ]
        embedding_cosine_similarity = F.cosine_similarity(community_centers_per_user, user_embedding, dim=-1)

        community_scale = self.mailbox["community_scale_{}".format(cur_community_type)]
        community_scale_per_user = community_scale[
            nodes.data["partition_{}".format(cur_community_type)].long()
        ]

        return {
            "community_belonging_{}".format(cur_community_type): embedding_cosine_similarity,
            "community_scale_{}".format(cur_community_type): community_scale_per_user,
        }

    def msg_update_daily_usage(self, edges):
        return {
            "edge_tag": edges.data["tag"],
        }

    def reduce_update_daily_usage(self, nodes):
        return {
            "daily_usage": torch.ones(nodes.mailbox["edge_tag"].shape[0]) * nodes.mailbox["edge_tag"].shape[1],
        }
