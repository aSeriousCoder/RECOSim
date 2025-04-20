import torch
import numpy as np
import pickle
from tqdm import tqdm


MAX_BATCH_SIZE = 512


class TopicGMM:
    def __init__(self):
        with open('Modules/ckpts/gmm_models.pkl', 'rb') as file:
            self.gmm_models = pickle.load(file)
            self.gmm_means = [torch.from_numpy(gmm.means_[0]) for gmm in self.gmm_models]
            self.gmm_covs = [torch.from_numpy(gmm.covariances_[0]) for gmm in self.gmm_models]
            self.gmm_cholesky = [torch.cholesky(cov) for cov in self.gmm_covs]

    def predict_topic(self, embeddings):
        # print('[TopicGMM] Predicting')
        pred = []
        num_batch = embeddings.shape[0] // MAX_BATCH_SIZE
        if embeddings.shape[0] % MAX_BATCH_SIZE > 0:
            num_batch += 1        
        for i in tqdm(range(num_batch)):
            emb = embeddings[i * MAX_BATCH_SIZE : (i + 1) * MAX_BATCH_SIZE]
            probs = torch.from_numpy(np.stack([gmm.score_samples(emb) for gmm in self.gmm_models]).T)
            pred.append(probs.softmax(dim=1).argmax(dim=1))
        return torch.concat(pred)
    
    def generate(self, topic_ids):
        """
        给定生成的话题序列topic_ids
        使用对应id位置的gmm_models生成embeddings
        """
        # print('[TopicGMM] Generating')
        eps = torch.randn([len(topic_ids), self.gmm_means[0].shape[-1]])
        generated_embeddings = [(self.gmm_means[topic_ids[i]] + self.gmm_cholesky[topic_ids[i]].double() @ eps[i].double()).float() for i in range(len(topic_ids))]
        return torch.stack(generated_embeddings)