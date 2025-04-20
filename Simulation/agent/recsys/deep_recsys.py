import torch
import dgl
import numpy as np
import pandas as pd
import faiss  
from Simulation.agent.recsys.base_recsys import BaseRecSys
from tqdm import tqdm

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import *
from recbole.trainer import Trainer
from recbole.utils import get_model, get_trainer
from recbole.data.interaction import Interaction


class DeepRecSys(BaseRecSys):
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
        self.build_dataset_for_feature_model()
        self.train_feature_model()
        self.build_dataset_for_ranking_model()
        self.train_ranking_model()
    
    def build_dataset_for_feature_model(self):
        '''
        Generate data files for Bole training
        .inter file format:
            | user_id:token | item_id:token | rating:float |
        '''
        self._logger.debug("Building dataset for feature model ...")
        export_dataset_dir = self._config.bole_post_data_dir
        # window中采样
        self.msg_sender.mailbox['begin_day'] = self._graph._cur_tag - self._config.window_size
        self.msg_sender.mailbox['end_day'] = self._graph._cur_tag
        active_post_musk = (self._graph.graph.edges['create_post'].data['tag'] >= self.msg_sender.mailbox['begin_day']) & (self._graph.graph.edges['create_post'].data['tag'] < self.msg_sender.mailbox['end_day'])
        interaction_sub_g = dgl.node_subgraph(self._graph.graph, {
            'user': torch.tensor([True] * self._graph.graph.num_nodes('user')),
            'post': active_post_musk,
        }, store_ids=True)  # the labels of post are reset
        user_like_post = interaction_sub_g.adj('like').indices()
        user_comment_on_post = interaction_sub_g.adj('comment_on').indices()
        user_repost_on_post = interaction_sub_g.adj('repost_on').indices()
        user_interact_post = torch.cat((user_like_post.T, user_comment_on_post.T, user_repost_on_post.T), dim=0)
        user_interact_post[:,1] = interaction_sub_g.nodes['post'].data['_ID'][user_interact_post[:,1]]  # convert to original ID
        self.trained_user_ids = user_interact_post[:, 0].unique()
        self.trained_post_ids = user_interact_post[:, 1].unique()
        # Write Data
        self._logger.debug("Number of interactions: {}".format(user_interact_post.shape[0]))
        interaction_data = user_interact_post.numpy().astype(str).tolist()
        interaction_columns = ['user_id:token', 'item_id:token']
        interaction_columns_str  = '\t'.join(interaction_columns)
        interaction_data_str = '\n'.join(['\t'.join(record) for record in interaction_data])
        interaction_str = f"{interaction_columns_str}\n{interaction_data_str}"
        interaction_path = f"{export_dataset_dir}/SSN_P.inter"
        with open(interaction_path, 'w') as f:
            f.write(interaction_str)

    def train_feature_model(self):
        self._logger.debug("Training feature model ...")
        feature_model_list = self._config.feature_model_list.split(',')
        for feature_model in feature_model_list:
            self._logger.debug("Training feature model - {} ...".format(feature_model))
            config = Config(model=feature_model, config_file_list=[self._config.feature_model_training_configs])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
            model = get_model(config.model)(config, dataset).to(config.device)
            trainer = get_trainer(config.MODEL_TYPE, config.model)(config, model)
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, verbose=True, saved=True, show_progress=True
            )
            self._logger.debug('best valid score: {}'.format(best_valid_score))
            self._logger.debug('best valid result: {}'.format(best_valid_result))
            user_id_mapping = torch.from_numpy(dataset.field2id_token['user_id'][1:].astype(int)).long()
            item_id_mapping = torch.from_numpy(dataset.field2id_token['item_id'][1:].astype(int)).long()
            user_embedding = model.user_embedding.weight.detach().cpu()[1:]
            item_embedding = model.item_embedding.weight.detach().cpu()[1:]
            self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model)] = torch.zeros(self._graph.graph.num_nodes('user'), user_embedding.shape[1])
            self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model)][user_id_mapping] = user_embedding
            self._graph.graph.nodes['post'].data['feature_{}'.format(feature_model)] = torch.zeros(self._graph.graph.num_nodes('post'), item_embedding.shape[1])
            self._graph.graph.nodes['post'].data['feature_{}'.format(feature_model)][item_id_mapping] = item_embedding

    def build_dataset_for_ranking_model(self):
        self._logger.debug("Building dataset for ranking model ...")
        self._logger.debug("Building post dataset ...")
        export_dataset_dir = self._config.bole_post_data_dir
        feature_model_list = self._config.feature_model_list.split(',')
        user_columns = ['user_id:token'] + ['feature_{}:float_seq'.format(feature_model) for feature_model in feature_model_list]
        user_data = [np.expand_dims(self._graph.graph.nodes('user').numpy().astype(str), axis=1)]
        for feature_model in feature_model_list:
            feats = self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model)].numpy().astype(str)
            feats = np.expand_dims(np.array([' '.join(feat) for feat in feats]), axis=1)
            user_data.append(feats)
        user_data = np.concatenate(user_data, axis=1).tolist()
        user_columns_str  = '\t'.join(user_columns)
        user_data_str = '\n'.join(['\t'.join(record) for record in user_data])
        user_str = f"{user_columns_str}\n{user_data_str}"
        user_path = f"{export_dataset_dir}/SSN_P.user"
        with open(user_path, 'w') as f:
            f.write(user_str)
        
        post_columns = ['item_id:token'] + ['feature_{}:float_seq'.format(feature_model) for feature_model in feature_model_list]
        post_data = [np.expand_dims(self._graph.graph.nodes('post').numpy().astype(str), axis=1)]
        for feature_model in feature_model_list:
            feats = self._graph.graph.nodes['post'].data['feature_{}'.format(feature_model)].numpy().astype(str)
            feats = np.expand_dims(np.array([' '.join(feat) for feat in feats]), axis=1)  # time consuming !!!
            post_data.append(feats)
        post_data = np.concatenate(post_data, axis=1).tolist()
        post_columns_str  = '\t'.join(post_columns)
        post_data_str = '\n'.join(['\t'.join(record) for record in post_data])
        post_str = f"{post_columns_str}\n{post_data_str}"
        post_path = f"{export_dataset_dir}/SSN_P.item"
        with open(post_path, 'w') as f:
            f.write(post_str)

        self._logger.debug("Building friend dataset ...")
        # "item" is user as well 
        export_dataset_dir = self._config.bole_friend_data_dir
        user_path = f"{export_dataset_dir}/SSN_F.user"
        with open(user_path, 'w') as f:
            f.write(user_str)
        user_path = f"{export_dataset_dir}/SSN_F.item"
        user_str = 'item' + user_str[4:]
        with open(user_path, 'w') as f:
            f.write(user_str)
        user_following_relationship = self._graph.graph.adj('follow').indices()
        follow_data = user_following_relationship.T.numpy().astype(str).tolist()
        self._logger.debug("Number of follow relationships: {}".format(len(follow_data)))
        follow_columns = ['user_id:token', 'item_id:token']
        follow_columns_str  = '\t'.join(follow_columns)
        follow_data_str = '\n'.join(['\t'.join(record) for record in follow_data])
        follow_str = f"{follow_columns_str}\n{follow_data_str}"
        follow_path = f"{export_dataset_dir}/SSN_F.inter"
        with open(follow_path, 'w') as f:
            f.write(follow_str)
          
    def train_ranking_model(self):
        self._logger.debug("Training ranking model ...")
        ranking_model = self._config.ranking_model
        # Post Ranking Model
        config = Config(model=ranking_model, config_file_list=[self._config.post_ranking_model_training_configs])
        dataset = create_dataset(config)  # time consuming !!!
        train_data, valid_data, test_data = data_preparation(config, dataset)
        post_ranking_model = get_model(config.model)(config, dataset).to(config.device)
        trainer = get_trainer(config.MODEL_TYPE, config.model)(config, post_ranking_model)
        best_valid_score, best_valid_result = trainer.fit(  # time consuming !!!
            train_data, valid_data, verbose=True, saved=True, show_progress=True
        )
        self._logger.debug('best valid score: {}'.format(best_valid_score))
        self._logger.debug('best valid result: {}'.format(best_valid_result))

        config = Config(model=ranking_model, config_file_list=[self._config.friend_ranking_model_training_configs])
        dataset = create_dataset(config)  # time consuming !!!
        train_data, valid_data, test_data = data_preparation(config, dataset)
        friend_ranking_model = get_model(config.model)(config, dataset).to(config.device)
        trainer = get_trainer(config.MODEL_TYPE, config.model)(config, friend_ranking_model)
        best_valid_score, best_valid_result = trainer.fit(  # time consuming !!!
            train_data, valid_data, verbose=True, saved=True, show_progress=True
        )
        self._logger.debug('best valid score: {}'.format(best_valid_score))
        self._logger.debug('best valid result: {}'.format(best_valid_result))

        self.post_ranking_model = post_ranking_model
        self.friend_ranking_model = friend_ranking_model
        self.ranking_model_device = config.device

    def post_batch_ranking(self, user_ids, post_ids):
        batch = Interaction({
            'user_id': user_ids,
            'item_id': post_ids,
        }).to(self.ranking_model_device)
        return self.post_ranking_model.predict(batch).detach().cpu()
    
    def friend_batch_ranking(self, src_user_ids, dst_user_ids):
        batch = Interaction({
            'user_id': src_user_ids,
            'item_id': dst_user_ids,
        }).to(self.ranking_model_device)
        return self.friend_ranking_model.predict(batch).detach().cpu()

    def post_recall(self):
        '''
        For each user, recall add to 10*_post_rec_list_length posts from different feature space
        相似向量搜索问题
        '''
        self._logger.debug("Recalling posts ...")
        recall_scale = 10 * self._post_rec_list_length  # 1000 posts per user
        feature_model_list = self._config.feature_model_list.split(',')  # 手动保证整除关系 (1,2,4,5,10...)
        recall_scale_per_feature_model = recall_scale // len(feature_model_list)
        trained_user_musk = self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model_list[0])].sum(dim=1) != 0
        trained_post_musk = self._graph.graph.nodes['post'].data['feature_{}'.format(feature_model_list[0])].sum(dim=1) != 0
        trained_user_index = np.array([i for i in range(self._graph.graph.num_nodes('user')) if trained_user_musk[i]])
        trained_post_index = np.array([i for i in range(self._graph.graph.num_nodes('post')) if trained_post_musk[i]])

        trained_user_post_recall_results = []
        for feature_model in feature_model_list:
            n_unit = self._config.faiss_n_unit  # 单元数
            k = recall_scale_per_feature_model  # 查询向量个数
            emb_dim = self._graph.graph.nodes['post'].data['feature_{}'.format(feature_model)].shape[1]
            quantizer = faiss.IndexFlatL2(emb_dim)  # 设置量化器建立检索空间
            index = faiss.IndexIVFFlat(quantizer, emb_dim, n_unit)
            index.train(self._graph.graph.nodes['post'].data['feature_{}'.format(feature_model)][trained_post_musk].numpy())  # 训练检索库
            index.add(self._graph.graph.nodes['post'].data['feature_{}'.format(feature_model)][trained_post_musk].numpy())
            index.nprobe = self._config.faiss_n_probe  # 设置在多少个相近单元进行查找
            recall_score, recall_index = index.search(self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model)][trained_user_musk].numpy(), k) 
            recall_index = trained_post_index[recall_index]
            trained_user_post_recall_results.append(torch.from_numpy(recall_index).long())
        trained_user_post_recall_results = torch.cat(trained_user_post_recall_results, dim=1)

        trained_user_post_recall_results_in_pair = torch.cat([
            torch.cat([torch.from_numpy(trained_user_index).unsqueeze(1)] * recall_scale, dim=1).reshape(-1, 1),
            trained_user_post_recall_results.reshape(-1, 1),
        ], dim=1).int()

        return trained_user_post_recall_results_in_pair
    
    def friends_recall(self):
        '''
        For each user, recall add to 10*_friend_rec_list_length friends from different feature space
        相似向量搜索问题
        '''
        self._logger.debug("Recalling friends ...")
        recall_scale = 10 * self._friend_rec_list_length  # 100 posts per user
        feature_model_list = self._config.feature_model_list.split(',')  # 手动保证整除关系 (1,2,4,5,10...)
        recall_scale_per_feature_model = recall_scale // len(feature_model_list)
        trained_user_musk = self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model_list[0])].sum(dim=1) != 0
        trained_user_index = np.array([i for i in range(self._graph.graph.num_nodes('user')) if trained_user_musk[i]])

        trained_user_friends_recall_results = []
        for feature_model in feature_model_list:
            n_unit = self._config.faiss_n_unit  # 单元数
            k = recall_scale_per_feature_model  # 查询向量个数
            emb_dim = self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model)].shape[1]
            quantizer = faiss.IndexFlatL2(emb_dim)  # 设置量化器建立检索空间
            index = faiss.IndexIVFFlat(quantizer, emb_dim, n_unit)
            index.train(self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model)][trained_user_musk].numpy())  # 训练检索库
            index.add(self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model)][trained_user_musk].numpy())
            index.nprobe = self._config.faiss_n_probe  # 设置在多少个相近单元进行查找
            recall_score, recall_index = index.search(self._graph.graph.nodes['user'].data['feature_{}'.format(feature_model)][trained_user_musk].numpy(), k) 
            recall_index = trained_user_index[recall_index]
            trained_user_friends_recall_results.append(torch.from_numpy(recall_index).long())
        trained_user_friends_recall_results = torch.cat(trained_user_friends_recall_results, dim=1)  # in mat

        trained_user_friends_recall_results_in_pair = torch.cat([
            torch.cat([torch.from_numpy(trained_user_index).unsqueeze(1)] * recall_scale, dim=1).reshape(-1, 1),
            trained_user_friends_recall_results.reshape(-1, 1),
        ], dim=1).int()

        return trained_user_friends_recall_results_in_pair

    def post_ranking(self, post_recall_results):
        '''
        post_recall_results: [user_id, post_id] * N
        10*_post_rec_list_length -> _post_rec_list_length
        filter out posts that have been interacted
        '''
        self._logger.debug("Ranking posts ...")
        num_batch = post_recall_results.shape[0] // self._config.ranking_batch_size + 1
        ranking_scores = torch.zeros(post_recall_results.shape[0])
        for i in tqdm(range(num_batch), disable=not self._config.show_progress):
            batch = post_recall_results[i * self._config.ranking_batch_size : (i + 1) * self._config.ranking_batch_size]
            user_ids = batch[:, 0]
            post_ids = batch[:, 1]
            post_scores = self.post_batch_ranking(user_ids, post_ids)
            ranking_scores[i * self._config.ranking_batch_size : (i + 1) * self._config.ranking_batch_size] = post_scores

        filter_out_mat_1 = self._graph.graph.has_edges_between(post_recall_results[:,0].int(), post_recall_results[:,1].int(), ('user', 'post_rec', 'post'))
        filter_out_mat_2 = self._graph.graph.has_edges_between(post_recall_results[:,0].int(), post_recall_results[:,1].int(), ('user', 'create_post', 'post'))
        in_init_musk = post_recall_results[:, 1] < self.interaction_sub_g.num_nodes('post')
        filter_out_mat_3 = torch.zeros(post_recall_results.shape[0]).bool()
        filter_out_mat_3[in_init_musk] = self.interaction_sub_g.has_edges_between(post_recall_results[in_init_musk][:,0].int(), post_recall_results[in_init_musk][:,1].int(), ('user', 'like', 'post'))
        filter_out_mat_4 = torch.zeros(post_recall_results.shape[0]).bool()
        filter_out_mat_4[in_init_musk] = self.interaction_sub_g.has_edges_between(post_recall_results[in_init_musk][:,0].int(), post_recall_results[in_init_musk][:,1].int(), ('user', 'comment_on', 'post'))
        filter_out_mat_5 = torch.zeros(post_recall_results.shape[0]).bool()
        filter_out_mat_5[in_init_musk] = self.interaction_sub_g.has_edges_between(post_recall_results[in_init_musk][:,0].int(), post_recall_results[in_init_musk][:,1].int(), ('user', 'repost_on', 'post'))
        filter_out_mat = filter_out_mat_1 | filter_out_mat_2 | filter_out_mat_3 | filter_out_mat_4 | filter_out_mat_5
        ranking_scores[filter_out_mat.int() == 1] = 0

        top_ranked = torch.topk(ranking_scores.reshape(-1, self._post_rec_list_length*10), dim=1, k=self._post_rec_list_length).indices
        post_ranking_result_in_mat = post_recall_results[:, 1].reshape(-1, self._post_rec_list_length*10).gather(1, top_ranked)
        trained_user_id = post_recall_results[:,0].unique()
        post_ranking_result_in_pair = torch.cat([
            torch.cat([trained_user_id.unsqueeze(1)] * self._post_rec_list_length, dim=1).reshape(-1, 1),
            post_ranking_result_in_mat.reshape(-1, 1),
        ], dim=1)
        
        self._logger.debug("Max Post_ID in Post Ranking Results: {}".format(post_ranking_result_in_pair[:,1].max()))
        return post_ranking_result_in_pair

    def friends_ranking(self, friends_recall_results):
        '''
        10*_friend_rec_list_length -> _friend_rec_list_length
        filter out friends that have been followed
        '''
        self._logger.debug("Ranking friends ...")
        num_batch = friends_recall_results.shape[0] // self._config.ranking_batch_size + 1
        ranking_scores = torch.zeros(friends_recall_results.shape[0])
        for i in tqdm(range(num_batch), disable=not self._config.show_progress):
            batch = friends_recall_results[i * self._config.ranking_batch_size : (i + 1) * self._config.ranking_batch_size]
            src_user_ids = batch[:, 0]
            dst_user_ids = batch[:, 1]
            friends_scores = self.friend_batch_ranking(src_user_ids, dst_user_ids)
            ranking_scores[i * self._config.ranking_batch_size : (i + 1) * self._config.ranking_batch_size] = friends_scores

        filter_out_mat_1 = self._graph.graph.has_edges_between(friends_recall_results[:,0].int(), friends_recall_results[:,1].int(), ('user', 'friend_rec', 'user'))
        filter_out_mat_2 = self.follow_sub_g.has_edges_between(friends_recall_results[:,0].int(), friends_recall_results[:,1].int(), ('user', 'follow', 'user'))
        filter_out_mat = filter_out_mat_1 | filter_out_mat_2
        ranking_scores[filter_out_mat == 1] = 0

        top_ranked = torch.topk(ranking_scores.reshape(-1, self._friend_rec_list_length*10), dim=1, k=self._friend_rec_list_length).indices
        friends_ranking_result_in_mat = friends_recall_results[:, 1].reshape(-1, self._friend_rec_list_length*10).gather(1, top_ranked)
        trained_user_id = friends_recall_results[:,0].unique()
        friends_ranking_result_in_pair = torch.cat([
            torch.cat([trained_user_id.unsqueeze(1)] * self._friend_rec_list_length, dim=1).reshape(-1, 1),
            friends_ranking_result_in_mat.reshape(-1, 1),
        ], dim=1)
        
        self._logger.debug("Max User_ID in User Ranking Results: {}".format(friends_ranking_result_in_pair[:,1].max()))
        return friends_ranking_result_in_pair

    def recommend_post(self):
        self._logger.debug("Recommending news ...")
        return self.post_ranking(self.post_recall())

    def recommend_friends(self):
        self._logger.debug("Recommending friends ...")
        return self.friends_ranking(self.friends_recall())


class MsgSender:
    def __init__(self):
        self.mailbox = {}

    def interactions_in_window(self, edges):
        begin_day = self.mailbox["begin_day"]
        end_day = self.mailbox["end_day"]
        edge_tag = edges.data['tag']
        return (edge_tag >= begin_day) & (edge_tag < end_day)
