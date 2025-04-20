import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from Modules.scoring_model.datasets import build_datasets
from Modules.scoring_model.user_interact_score_model import UserInteractScoreModel
from Modules.scoring_model.user_follow_score_model import UserFollowScoreModel


def train_UserScoreModel(action_type):
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="4,")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--validation_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    args = parser.parse_args()
    # data
    train_dataset, valid_dataset, test_dataset = build_datasets(prefix=action_type)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.validation_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)
    # model
    if action_type != 'follow':
        model = UserInteractScoreModel(args, hidden_dim=512)
    else:
        model = UserFollowScoreModel(args, hidden_dim=512)
    # train model
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    with open(f'Modules/ckpts/scoring_model_{action_type}.pkl', 'wb') as f:
        pickle.dump(model, f)


def get_quantile_values(action_type):
    print(f'Get quantile values for {action_type} scoring model')
    with open(f'Modules/ckpts/scoring_model_{action_type}.pkl', 'rb') as f:
        model = pickle.load(f)
    train_dataset, test_dataset, valid_dataset = build_datasets(prefix=action_type)
    all_score = []
    for dataset in [train_dataset, test_dataset, valid_dataset]:
        dataloader = DataLoader(dataset, batch_size=64)
        for batch_data in tqdm(dataloader):
            if action_type != 'follow':
                user_history, user_embedding, post_embedding, _ = batch_data
                score = model(user_history, user_embedding, post_embedding)
                all_score.append(score.detach().cpu())
            else:
                user_history, user_embedding, friend_history, friend_embedding, _, _ = batch_data
                score = model(user_history, user_embedding, friend_history, friend_embedding)
                all_score.append(score.detach().cpu())
    all_score = torch.cat(all_score, dim=0)
    quantile_values = torch.stack([torch.quantile(all_score, i/100) for i in range(0,101)])
    with open(f'Modules/ckpts/quantile_values_{action_type}.pkl', 'wb') as f:
        pickle.dump(quantile_values, f)


def main():
    train_UserScoreModel(action_type='repost')
    train_UserScoreModel(action_type='comment')
    train_UserScoreModel(action_type='like')
    train_UserScoreModel(action_type='follow')
    get_quantile_values(action_type='repost')
    get_quantile_values(action_type='comment')
    get_quantile_values(action_type='like')
    get_quantile_values(action_type='follow')
    print("Done!")
