import os
import torch
from torch import nn
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm


DATASET_SAVING_DIR = "Modules/data/"
MAX_EPOCHS = 10
TRAIN_RATIO = 0.8


def main():
    ugt_states = []
    action_densities = []
    for i in range(1007, 1014):
        raw_data = torch.load(DATASET_SAVING_DIR + "acticity_density_" + str(i) + ".pt")
        ugt_states.append(raw_data[:, :-4])
        action_densities.append(raw_data[:, -4:])

    action_types = ['post', 'repost', 'comment', 'like']
    indexs = [0, 1, 2, 3]

    for action_type, index in zip(action_types, indexs):

        feat = torch.cat(ugt_states[1:], dim=0) - torch.cat(ugt_states[:-1], dim=0)
        label = torch.cat(action_densities[1:], dim=0)[:, index] - torch.cat(action_densities[:-1], dim=0)[:, index]
        label_value = torch.cat(action_densities[1:], dim=0)[:, index]  # the absolute value

        all_size, feat_dim = feat.shape
        train_size = int(all_size * TRAIN_RATIO)

        reg_model = LinearRegression().fit(feat[:train_size].numpy(), label[:train_size].numpy())
        pred = torch.from_numpy(reg_model.predict(feat[train_size:].numpy()))
        truth = label[train_size:]
        abs_truth = label_value[train_size:]

        with open(f'Modules/ckpts/activity_model_{action_type}.pkl', 'wb') as f:
            pickle.dump(reg_model, f)

        diff = (pred - truth).abs()
        N = diff.shape[0]
        print("Hit@0: ", (diff < 0.5).sum().item() / N)
        print("Hit@1: ", (diff < 1.5).sum().item() / N)
        print("Hit@2: ", (diff < 2.5).sum().item() / N)
        print("Hit@5: ", (diff < 5.5).sum().item() / N)

        high_activity_musk = abs_truth > 10
        diff = (pred[high_activity_musk] - truth[high_activity_musk]).abs()
        N = diff.shape[0]
        print("Hit@0: ", (diff < 0.5).sum().item() / N)
        print("Hit@1: ", (diff < 1.5).sum().item() / N)
        print("Hit@2: ", (diff < 2.5).sum().item() / N)
        print("Hit@5: ", (diff < 5.5).sum().item() / N)
        print("Hit@10: ", (diff < 10.5).sum().item() / N)
        print("Hit@20: ", (diff < 20.5).sum().item() / N)

        feat = torch.stack(ugt_states[1:], dim=0) - torch.stack(ugt_states[:-1], dim=0)
        label = torch.stack(action_densities[1:], dim=0)[:, :, index] - torch.stack(action_densities[:-1], dim=0)[:, :, index]

        user_action_ugt_mutual_informations = []
        for uid in tqdm(range(feat.shape[1])):
            mi = mutual_info_regression(feat[:, uid, :], label[:, uid])
            user_action_ugt_mutual_informations.append(mi.max())

        with open(f'Modules/ckpts/activity_mi_{action_type}.pkl', 'wb') as f:
            pickle.dump(torch.tensor(user_action_ugt_mutual_informations), f)

        print('shape', torch.tensor(user_action_ugt_mutual_informations).shape)
        print('mean', torch.tensor(user_action_ugt_mutual_informations).mean())
        print('max', torch.tensor(user_action_ugt_mutual_informations).max())
        print('min', torch.tensor(user_action_ugt_mutual_informations).min())






