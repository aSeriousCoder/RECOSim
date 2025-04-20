import os
import torch
import numpy as np


NUM_NEGATIVE_SAMPLES = 5


def build_datasets(prefix):
    raw_data_dir =  "Modules/data"
    data = []
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith(prefix):
            print("Processing file: {}".format(filename))
            data.append(torch.load(f))
    data = torch.cat(data, dim=0)
    train_size = int(len(data) * 0.8)
    valid_size = int(len(data) * 0.1)
    test_size = int(len(data) * 0.1)
    train_data = data[:train_size]
    valid_data = data[train_size:train_size+valid_size]
    test_data = data[train_size+valid_size:train_size+valid_size+test_size]
    return UniversalDataset(prefix, train_data), UniversalDataset(prefix, valid_data), UniversalDataset(prefix, test_data)


class UniversalDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, prefix, data):
        self.prefix = prefix
        shuffle_indices = torch.randperm(data.shape[0])
        self.data = data[shuffle_indices]
        shuffle_indices_nagative_sampling = [torch.randperm(data.shape[0]) for _ in range(NUM_NEGATIVE_SAMPLES)]
        if self.data.shape[1] == 22:
            self.user_history = self.data[:, :20, :]
            self.user_embedding = self.data[:, 20, :]
            self.positive_post_embedding = self.data[:, 21, :]
            self.negative_post_embedding = torch.concat([self.data[index, 21, :].unsqueeze(1) for index in shuffle_indices_nagative_sampling], dim=1)
        else:
            self.user_history = self.data[:, :20, :]
            self.user_embedding = self.data[:, 20, :]
            self.positive_friend_history = self.data[:, 21:41, :]
            self.positive_friend_embedding = self.data[:, 41, :]
            self.negative_friend_history = torch.concat([self.data[index, 21:41, :].unsqueeze(1) for index in shuffle_indices_nagative_sampling], dim=1)
            self.negative_friend_embedding = torch.concat([self.data[index, 41, :].unsqueeze(1) for index in shuffle_indices_nagative_sampling], dim=1)


    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        if self.prefix != 'follow':
            return self.user_history[index], self.user_embedding[index], self.positive_post_embedding[index], self.negative_post_embedding[index]
        else:
            return self.user_history[index], self.user_embedding[index], self.positive_friend_history[index], self.positive_friend_embedding[index], self.negative_friend_history[index], self.negative_friend_embedding[index]
