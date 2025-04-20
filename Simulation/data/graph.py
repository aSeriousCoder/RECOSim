import os
import json
import torch
import numpy as np
import pandas as pd
import dgl
import pickle
from tqdm import tqdm
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import save_info, load_info
from Data.run_spider import *
from Modules.generate_model.topic_gmm import TopicGMM


class SimuLineGraph(DGLDataset):
    def __init__(self, config, logger, name='SimuLineGraph', hash_key='SimuLineGraph', force_reload=False):
        self._config = config
        self._logger = logger
        self._name = name
        self._hash_key = hash_key
        self._force_reload = force_reload
        self._verbose = True
        self._graph = None  # Save & Load
        self._info = {}  # Save & Load
        self._raw_dir = 'Data/result'
        self._url = None
        self._save_dir = config.raw_graph_save_dir
        self._save_graph_path = f"{self._save_dir}/graph_base.json"
        self._save_info_path = f"{self._save_dir}/info_base.json"
        self._simulation_save_dir = config.simulation_result_dir
        self._history_length = config.history_length
        self._padding = config.padding
        self._cur_tag = config.beginning_tag
        self._window_size = config.window_size
        self._history_window_beginning = self._cur_tag - self._window_size
        self._topic_gmm = TopicGMM()

        if config.continue_simulation:
            self.load_simulation(tag=config.continue_tag)
        else:
            self._load()
        
        if self._config.build_dataset:  # all tag get padded
            for etype in self.graph.etypes:
                if 'tag' in self.graph.edges[etype].data.keys():
                    self.graph.edges[etype].data['tag'] += self._padding
        else:  # only tag in the window get padded
            for etype in self.graph.etypes:
                if 'tag' in self.graph.edges[etype].data.keys():
                    in_window_musk = self.graph.edges[etype].data['tag'] < self._window_size
                    self.graph.edges[etype].data['tag'][in_window_musk] += self._padding

    def process(self):
        '''
        # - History: '2023-05-20 00:00:00' - '2023-05-27 00:00:00' as features - 7 days - index 0~6
        # - Train: '2023-05-27 00:00:00' - '2023-06-01 00:00:00' as train_labels - 5 days - index 7~11
        # - Test: '2023-06-01 00:00:00' - '2023-06-03 00:00:00' as test_labels - 2 days - index 12~13
        # - Valid: '2023-06-03 00:00:00' - '2023-06-06 00:00:00' as valid_labels - 3 days - index 14~16

        Node types: user, post, repost, comment
        Link types:
            (user, follow, user),
            (user, create_post, post),
            (user, create_repost, repost),
            (repost, repost_of, post),
            (user, create_comment, comment),
            (comment, comment_of, post),
            (user, like, post),
        
        Link attributes:  # for Post/Repost/Comment/Like
            tag: 0~15-data 9xxx-simulated
        '''

        self._logger.debug("Reading data ...")
        participating_user_id_list = read_user_id_list_from_topic_participate()
        user_following_relations = read_user_following_relations()
        user_info = read_user_info()
        post_id_list, post_info, user_posting_relations, user_reposting_relations = read_user_posts()
        comment_id_list, comment_info, user_comment_relations = read_post_comments()
        like_info, user_liking_relations = read_user_likes()
        post_embedding_array = np.load("Data/result/post_embeddings.npy")
        comment_embedding_array = np.load("Data/result/comment_embeddings.npy")

        self._logger.debug("Processing nodes & links ...")
        user_list = participating_user_id_list  # uid -> user_id in weibo
        post_list = [post_id for user_id, post_id in tqdm(user_posting_relations, disable=not self._config.show_progress) if user_id in user_list]
        repost_list = [repost_id for user_id, repost_id, raw_post in tqdm(user_reposting_relations, disable=not self._config.show_progress) if user_id in user_list]
        comment_list = [comment_id for user_id, comment_id, raw_post in tqdm(user_comment_relations, disable=not self._config.show_progress) if user_id in user_list]
        self._info['node_id_lists'] = {}
        self._info['node_id_lists']['user'] = user_list
        self._info['node_id_lists']['post'] = post_list
        self._info['node_id_lists']['repost'] = repost_list
        self._info['node_id_lists']['comment'] = comment_list
        self._info['num_nodes'] = {}
        self._info['num_nodes']['user'] = len(user_list)
        self._info['num_nodes']['post'] = len(post_list)
        self._info['num_nodes']['repost'] = len(repost_list)
        self._info['num_nodes']['comment'] = len(comment_list)

        user_id_2_uid = {user_id: uid for uid, user_id in enumerate(user_list)}
        post_id_2_pid = {post_id: pid for pid, post_id in enumerate(post_list)} 
        repost_id_2_rid = {repost_id: rid for rid, repost_id in enumerate(repost_list)}
        comment_id_2_cid = {comment_id: cid for cid, comment_id in enumerate(comment_list)}

        # links
        user_follow_user = [[user_id_2_uid[user_id], user_id_2_uid[following_user_id]] for user_id, following_user_id in tqdm(user_following_relations, disable=not self._config.show_progress)]
        user_create_post = [[user_id_2_uid[user_id], post_id_2_pid[post_id]] for user_id, post_id in tqdm(user_posting_relations, disable=not self._config.show_progress) if user_id in user_list]
        user_create_repost = [[user_id_2_uid[user_id], repost_id_2_rid[repost_id]] for user_id, repost_id, raw_post in tqdm(user_reposting_relations, disable=not self._config.show_progress) if  user_id in user_list and raw_post in post_id_2_pid]
        repost_of_post = [[repost_id_2_rid[repost_id], post_id_2_pid[raw_post]] for user_id, repost_id, raw_post in tqdm(user_reposting_relations, disable=not self._config.show_progress) if user_id in user_list and raw_post in post_id_2_pid]
        user_reposted_post = [[user_id_2_uid[user_id], post_id_2_pid[raw_post]] for user_id, repost_id, raw_post in tqdm(user_reposting_relations, disable=not self._config.show_progress) if user_id in user_list and raw_post in post_id_2_pid]
        user_create_comment = [[user_id_2_uid[user_id], comment_id_2_cid[comment_id]] for user_id, comment_id, raw_post in tqdm(user_comment_relations, disable=not self._config.show_progress) if user_id in user_list and raw_post in post_id_2_pid]
        comment_of_post = [[comment_id_2_cid[comment_id], post_id_2_pid[raw_post]] for user_id, comment_id, raw_post in tqdm(user_comment_relations, disable=not self._config.show_progress) if user_id in user_list and raw_post in post_id_2_pid]
        user_commented_post = [[user_id_2_uid[user_id], post_id_2_pid[raw_post]] for user_id, comment_id, raw_post in tqdm(user_comment_relations, disable=not self._config.show_progress) if user_id in user_list and raw_post in post_id_2_pid]
        user_like_post = [[user_id_2_uid[user_id], post_id_2_pid[post_id]] for user_id, post_id in tqdm(user_liking_relations, disable=not self._config.show_progress) if post_id in post_id_2_pid  and user_id in user_list]

        # link attributes - tag
        time_user_create_post = [post_info[post_id]['created_at'] for user_id, post_id in tqdm(user_posting_relations, disable=not self._config.show_progress) if user_id in user_list]
        time_user_create_repost = [post_info[repost_id]['created_at'] for user_id, repost_id, raw_post in tqdm(user_reposting_relations, disable=not self._config.show_progress) if  user_id in user_list and raw_post in post_id_2_pid]
        time_user_reposted_post = [post_info[repost_id]['created_at'] for user_id, repost_id, raw_post in tqdm(user_reposting_relations, disable=not self._config.show_progress) if  user_id in user_list and raw_post in post_id_2_pid]
        time_user_create_comment = [comment_info[comment_id]['created_at'] for user_id, comment_id, raw_post in tqdm(user_comment_relations, disable=not self._config.show_progress) if user_id in user_list and raw_post in post_id_2_pid]
        time_user_commented_post = [comment_info[comment_id]['created_at'] for user_id, comment_id, raw_post in tqdm(user_comment_relations, disable=not self._config.show_progress) if user_id in user_list and raw_post in post_id_2_pid]
        time_user_like_post = [like_info['{}-{}'.format(user_id, post_info[post_id]['mblogid'])]['created_at'] for user_id, post_id in tqdm(user_liking_relations, disable=not self._config.show_progress) if post_id in post_id_2_pid and user_id in user_list]

        def time_2_tag(time_list):
            tag_list = []
            for t in time_list:
                mm = int(t[5:7])
                dd = int(t[8:10])
                if mm == 6:
                    dd += 31
                if dd < 0:
                    dd = 20
                if dd > 35:
                    dd = 35
                tag_list.append(dd - 20)
            return tag_list
        
        tags = {
            'create_post': torch.Tensor(time_2_tag(time_user_create_post)),
            'create_repost': torch.Tensor(time_2_tag(time_user_create_repost)),
            'repost_on': torch.Tensor(time_2_tag(time_user_reposted_post)),
            'create_comment': torch.Tensor(time_2_tag(time_user_create_comment)),
            'comment_on': torch.Tensor(time_2_tag(time_user_commented_post)),
            'like': torch.Tensor(time_2_tag(time_user_like_post)),
            'follow': torch.zeros(len(user_follow_user)),
        }

        _links = {
            ('user', 'follow', 'user'): user_follow_user,
            ('user', 'create_post', 'post'): user_create_post,
            ('user', 'create_repost', 'repost'): user_create_repost,
            ('repost', 'repost_of', 'post'): repost_of_post,
            ('user', 'repost_on', 'post'): user_reposted_post,
            ('user', 'create_comment', 'comment'): user_create_comment,
            ('comment', 'comment_of', 'post'): comment_of_post,
            ('user', 'comment_on', 'post'): user_commented_post,
            ('user', 'like', 'post'): user_like_post,
            ('user', 'post_rec', 'post'): [],
            ('user', 'friend_rec', 'user'): [],
            ('user', 'history_follow', 'user'): [],
        }
        self._info['reverse_etypes'] = {}
        cur_relations = list(_links.keys())
        for relation in cur_relations:
            reverse_relation = '{}_r'.format(relation[1])
            _links[(relation[2], reverse_relation, relation[0])] = [[p[1], p[0]] for p in _links[relation]]
            self._info['reverse_etypes'][relation[1]] = reverse_relation
            self._info['reverse_etypes'][reverse_relation] = relation[1]

        self._logger.debug('Building Graph ...')
        self._graph = dgl.heterograph(_links, num_nodes_dict=self._info['num_nodes'], idtype=torch.int32)

        self._logger.debug('Mounting embeddings ...')

        for link_type in tags:
            self._graph.edges[link_type].data['tag'] = tags[link_type]
            self._graph.edges['{}_r'.format(link_type)].data['tag'] = tags[link_type]

        post_and_repost_index = {post_id: pid for pid, post_id in enumerate(post_id_list)}
        post_index = [post_and_repost_index[post_id] for post_id in post_list]
        post_attr_embedding = post_embedding_array[np.array(post_index)]
        repost_index = [post_and_repost_index[repost_id] for repost_id in repost_list]
        repost_attr_embedding = post_embedding_array[np.array(repost_index)]
        comment_index_all = {comment_id: cid for cid, comment_id in enumerate(comment_id_list)}
        comment_index = [comment_index_all[comment_id] for comment_id in comment_list]
        comment_attr_embedding = comment_embedding_array[np.array(comment_index)]

        self._graph.nodes['post'].data['embedding'] = torch.from_numpy(post_attr_embedding)
        self._graph.nodes['repost'].data['embedding'] = torch.from_numpy(repost_attr_embedding)
        self._graph.nodes['comment'].data['embedding'] = torch.from_numpy(comment_attr_embedding)

        post_embedding = self._graph.nodes['post'].data['embedding']
        repost_embedding = self._graph.nodes['repost'].data['embedding']
        comment_embedding = self._graph.nodes['comment'].data['embedding']

        post_embeddings = self._graph.nodes['post'].data['embedding']
        repost_embeddings = self._graph.nodes['repost'].data['embedding']
        comment_embeddings = self._graph.nodes['comment'].data['embedding']

        post_topic = self._topic_gmm.predict_topic(post_embeddings)
        repost_topic = self._topic_gmm.predict_topic(repost_embeddings)
        comment_topic = self._topic_gmm.predict_topic(comment_embeddings)

        self._graph.nodes['post'].data['topic_id'] = post_topic
        self._graph.nodes['repost'].data['topic_id'] = repost_topic
        self._graph.nodes['comment'].data['topic_id'] = comment_topic

        action_types = ['post', 'repost', 'like', 'comment']
        for action_type in action_types:
            with open(f'Modules/ckpts/activity_mi_{action_type}.pkl', 'rb') as file:
                self._graph.nodes['user'].data[f'{action_type}_mi'] = torch.tensor(pickle.load(file), dtype=torch.float32)

    def save(self):
        r"""
        保存图和标签
        """
        self._logger.debug('Saving to cache')
        save_graphs(self._save_graph_path, [self._graph])
        save_info(self._save_info_path, self._info)

    def load(self):
        r"""
         从目录 `self.save_path` 里读取处理过的数据
        """
        self._logger.debug('Loading from cache')
        graphs, label_dict = load_graphs(self._save_graph_path)
        self._graph = graphs[0]
        self._info = load_info(self._save_info_path)

    def save_simulation(self, tag=None):
        r"""
        保存图和标签
        """
        self._logger.debug('Saving to cache')
        self._info['cur_tag'] = self._cur_tag
        if tag:
            self._simulation_save_graph_path = f"{self._simulation_save_dir}/graph_base_tag_{tag}.json"
            self._simulation_save_info_path = f"{self._simulation_save_dir}/info_base_tag_{tag}.json"
        else:
            self._simulation_save_graph_path = f"{self._simulation_save_dir}/graph_base.json"
            self._simulation_save_info_path = f"{self._simulation_save_dir}/info_base.json"
        save_graphs(self._simulation_save_graph_path, [self._graph])
        save_info(self._simulation_save_info_path, self._info)

    def load_simulation(self, tag=None):
        r"""
         从目录 `self.save_path` 里读取处理过的数据
        """
        self._logger.debug('Loading from cache')
        if tag:
            self._simulation_save_graph_path = f"{self._simulation_save_dir}/graph_base_tag_{tag}.json"
            self._simulation_save_info_path = f"{self._simulation_save_dir}/info_base_tag_{tag}.json"
        else:
            self._simulation_save_graph_path = f"{self._simulation_save_dir}/graph_base.json"
            self._simulation_save_info_path = f"{self._simulation_save_dir}/info_base.json"
        graphs, label_dict = load_graphs(self._simulation_save_graph_path)
        self._graph = graphs[0]
        self._info = load_info(self._simulation_save_info_path)
        self._cur_tag = self._info['cur_tag']
        
    def has_cache(self):
        # 检查在 `self.save_path` 里是否有处理过的数据文件
        return os.path.exists(self._save_graph_path) and os.path.exists(self._save_info_path)
    
    @property
    def graph(self):
        return self._graph
    
    @property
    def info(self):
        return self._info

    def update_graph(self, post_recommendation, friends_recommendation, user_browse_actions, user_follow_actions):
        self._logger.debug("Updating Graph ...")
        # Recommendation
        post_recommendation_in_pair = post_recommendation.int()
        self.graph.add_edges(u=post_recommendation_in_pair[:,0], v=post_recommendation_in_pair[:, 1], etype='post_rec', data={
            'tag': torch.ones(len(post_recommendation_in_pair)) * self._cur_tag,
        })
        self.graph.add_edges(u=post_recommendation_in_pair[:,1], v=post_recommendation_in_pair[:, 0], etype='post_rec_r', data={
            'tag': torch.ones(len(post_recommendation_in_pair)) * self._cur_tag,
        })
        friends_recommendation_in_pair = friends_recommendation.int()
        self.graph.add_edges(u=friends_recommendation_in_pair[:,0], v=friends_recommendation_in_pair[:, 1], etype='friend_rec', data={
            'tag': torch.ones(len(friends_recommendation_in_pair)) * self._cur_tag,
        })
        self.graph.add_edges(u=friends_recommendation_in_pair[:,1], v=friends_recommendation_in_pair[:, 0], etype='friend_rec_r', data={
            'tag': torch.ones(len(friends_recommendation_in_pair)) * self._cur_tag,
        })
        # New Post
        post_id_padding = self.graph.num_nodes('post')
        num_new_post = user_browse_actions['new_post_embedding'].shape[0]
        self.graph.add_nodes(num_new_post, ntype='post', data={  # self.graph.nodes["post"].data
            'embedding': torch.tensor(user_browse_actions['new_post_embedding']),
            'topic_id': user_browse_actions['new_post_topic_id'],
            'popularity': torch.zeros(num_new_post),
        })
        new_create_post_rel = torch.tensor([[uid, post_id_padding+i] for i, uid in enumerate(user_browse_actions['new_post_relations'])]).int()
        self.graph.add_edges(u=new_create_post_rel[:,0], v=new_create_post_rel[:, 1], etype='create_post', data={
            'tag': torch.ones(num_new_post) * self._cur_tag,
        })
        self.graph.add_edges(u=new_create_post_rel[:,1], v=new_create_post_rel[:, 0], etype='create_post_r', data={
            'tag': torch.ones(num_new_post) * self._cur_tag,
            'through_pass': torch.zeros(num_new_post),
        })
        # New Repost
        repost_id_padding = self.graph.num_nodes('repost')
        num_new_repost = user_browse_actions['new_repost_embedding'].shape[0]
        self.graph.add_nodes(num_new_repost, ntype='repost', data={  # self.graph.nodes["repost"].data
            'embedding': torch.tensor(user_browse_actions['new_repost_embedding']),
            'topic_id': user_browse_actions['new_repost_topic_id'],
        })
        new_create_repost_rel = torch.tensor([[uid, repost_id_padding+i] for i, (uid, nid) in enumerate(user_browse_actions['new_repost_relations'])]).int()
        self.graph.add_edges(u=new_create_repost_rel[:,0], v=new_create_repost_rel[:, 1], etype='create_repost', data={
            'tag': torch.ones(num_new_repost) * self._cur_tag,
        })
        self.graph.add_edges(u=new_create_repost_rel[:,1], v=new_create_repost_rel[:, 0], etype='create_repost_r', data={
            'tag': torch.ones(num_new_repost) * self._cur_tag,
            'through_pass': torch.zeros(num_new_repost),
        })
        new_repost_of_rel = torch.tensor([[repost_id_padding+i, nid] for i, (uid, nid) in enumerate(user_browse_actions['new_repost_relations'])]).int()
        self.graph.add_edges(u=new_repost_of_rel[:,0], v=new_repost_of_rel[:, 1], etype='repost_of')
        self.graph.add_edges(u=new_repost_of_rel[:,1], v=new_repost_of_rel[:, 0], etype='repost_of_r')
        new_repost_on_rel = torch.tensor(user_browse_actions['new_repost_relations']).int()
        self.graph.add_edges(u=new_repost_on_rel[:,0], v=new_repost_on_rel[:, 1], etype='repost_on', data={
            'tag': torch.ones(num_new_repost) * self._cur_tag,
        })
        self.graph.add_edges(u=new_repost_on_rel[:,1], v=new_repost_on_rel[:, 0], etype='repost_on_r', data={
            'tag': torch.ones(num_new_repost) * self._cur_tag,
            'through_pass': torch.zeros(num_new_repost),
        })
        # New Comment
        comment_id_padding = self.graph.num_nodes('comment')
        num_new_comment = user_browse_actions['new_comment_embedding'].shape[0]
        self.graph.add_nodes(num_new_comment, ntype='comment', data={  # self.graph.nodes["comment"].data
            'embedding': torch.tensor(user_browse_actions['new_comment_embedding']),
            'topic_id': user_browse_actions['new_comment_topic_id'],
        })
        new_create_comment_rel = torch.tensor([[uid, comment_id_padding+i] for i, (uid, nid) in enumerate(user_browse_actions['new_comment_relations'])]).int()
        self.graph.add_edges(u=new_create_comment_rel[:,0], v=new_create_comment_rel[:, 1], etype='create_comment', data={
            'tag': torch.ones(num_new_comment) * self._cur_tag,
        })
        self.graph.add_edges(u=new_create_comment_rel[:,1], v=new_create_comment_rel[:, 0], etype='create_comment_r', data={
            'tag': torch.ones(num_new_comment) * self._cur_tag,
            'through_pass': torch.zeros(num_new_comment),
        })
        new_comment_of_rel = torch.tensor([[comment_id_padding+i, nid] for i, (uid, nid) in enumerate(user_browse_actions['new_comment_relations'])]).int()
        self.graph.add_edges(u=new_comment_of_rel[:,0], v=new_comment_of_rel[:, 1], etype='comment_of')
        self.graph.add_edges(u=new_comment_of_rel[:,1], v=new_comment_of_rel[:, 0], etype='comment_of_r')
        new_comment_on_rel = torch.tensor(user_browse_actions['new_comment_relations']).int()
        self.graph.add_edges(u=new_comment_on_rel[:,0], v=new_comment_on_rel[:, 1], etype='comment_on', data={
            'tag': torch.ones(num_new_comment) * self._cur_tag,
        })
        self.graph.add_edges(u=new_comment_on_rel[:,1], v=new_comment_on_rel[:, 0], etype='comment_on_r', data={
            'tag': torch.ones(num_new_comment) * self._cur_tag,
            'through_pass': torch.zeros(num_new_comment),
        })
        # New Like
        new_like_rel = torch.tensor([[uid, nid] for uid, nid in user_browse_actions['new_like_relations']]).int()
        self.graph.add_edges(u=new_like_rel[:,0], v=new_like_rel[:, 1], etype='like', data={
            'tag': torch.ones(new_like_rel.shape[0]) * self._cur_tag,
        })
        self.graph.add_edges(u=new_like_rel[:,1], v=new_like_rel[:, 0], etype='like_r', data={
            'tag': torch.ones(new_like_rel.shape[0]) * self._cur_tag,
            'through_pass': torch.zeros(new_like_rel.shape[0]),
        })
        # New Unfollow
        new_unfollow_rel = torch.tensor([[uid, fid] for uid, fid in user_follow_actions['unfollow_relations']]).int()
        unfollow_edge_ids = self.graph.edge_ids(new_unfollow_rel[:,0], new_unfollow_rel[:,1], etype='follow')
        self.graph.add_edges(u=new_unfollow_rel[:,0], v=new_unfollow_rel[:,1], etype='history_follow', data={
            'follow_tag': self._graph.edges['follow'].data['tag'][unfollow_edge_ids],  # 当时关注的时间
            'tag': torch.ones(new_unfollow_rel.shape[0]) * self._cur_tag,  # 取消关注的时间
        })
        self.graph.add_edges(u=new_unfollow_rel[:,1], v=new_unfollow_rel[:,0], etype='history_follow_r', data={
            'follow_tag': self._graph.edges['follow'].data['tag'][unfollow_edge_ids],  # 当时关注的时间
            'tag': torch.ones(new_unfollow_rel.shape[0]) * self._cur_tag,  # 取消关注的时间
        })
        self.graph.remove_edges(unfollow_edge_ids, etype='follow')  # BUG: 会导致新增的边的tag为0，必须放在前面
        self.graph.remove_edges(unfollow_edge_ids, etype='follow_r')
        # New Follow
        new_follow_rel = torch.tensor([[uid, fid] for uid, fid in user_follow_actions['new_follow_relations']]).int()
        self.graph.add_edges(u=new_follow_rel[:,0], v=new_follow_rel[:, 1], etype='follow', data={
            'tag': torch.ones(new_follow_rel.shape[0]) * self._cur_tag,
        })
        self.graph.add_edges(u=new_follow_rel[:,1], v=new_follow_rel[:, 0], etype='follow_r', data={
            'tag': torch.ones(new_follow_rel.shape[0]) * self._cur_tag,
            'through_pass': torch.zeros(new_follow_rel.shape[0]),
        })

        self.update_tag()

    def update_tag(self):
        self._cur_tag += 1
        self._history_window_beginning = self._cur_tag - self._window_size
