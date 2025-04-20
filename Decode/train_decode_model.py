"""
Source code modified from https://github.com/HKUST-KnowComp/GEIA
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
import argparse
import sys
from tqdm import tqdm
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from decode_beam_search import beam_decode_sentence
from typing import List, Tuple
from error import NotImplementedError


MIN_TEXT_LENGTH = 20  # the minimum length of the text used for training


def load_data(path):
    """
    Load your data, return a list of tuples, each tuple contains a prompt and a text
    The prompt is the text which the generated text respond to (optional, can be empty)
    For example:
    [
        ("", "Python is a efficient programming language!"),
        ("I like Python!", "I like it too!"),
    ]
    """
    raise NotImplementedError("You need to implement this function")


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return self.sequence_cross_entropy_with_logits(
            logits, targets, mask, label_smoothing, reduce
        )

    def sequence_cross_entropy_with_logits(
        self, logits, targets, mask, label_smoothing, reduce
    ):
        # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
        """
        label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
        """
        # shape : (batch * sequence_length, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()

        if label_smoothing > 0.0:
            num_classes = logits.size(-1)
            smoothing_value = label_smoothing / float(num_classes)
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(
                -1, targets_flat, 1.0 - label_smoothing
            )
            smoothed_targets = one_hot_targets + smoothing_value
            negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(
                -1, keepdim=True
            )
        else:
            # shape : (batch * sequence_length, 1)
            negative_log_likelihood_flat = -torch.gather(
                log_probs_flat, dim=1, index=targets_flat
            )
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])
        # shape : (batch, sequence_length)
        loss = negative_log_likelihood * mask
        if reduce:
            # shape : (batch,)
            loss = loss.sum(1) / (mask.sum(1) + 1e-13)
            if reduce == "batch":
                # shape : scalar
                loss = loss.mean()
        return loss


class LinearProjection(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LinearProjection, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.seq(x)


class Corpus(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def collate(self, unpacked_data):
        return unpacked_data


def train(data, config):
    # =============================
    device = config["device"]
    batch_size = config["batch_size"]
    model = SentenceTransformer(config["embed_model"], device=device)  # dim 512
    dataset = Corpus(data)
    dataloader = DataLoader(
        dataset=dataset, shuffle=True, batch_size=batch_size, collate_fn=dataset.collate
    )
    print("Train data loaded")
    # =============================
    projection = LinearProjection(in_dim=512, hidden_dim=256, out_dim=768).to(device)
    model_decoder = AutoModelForCausalLM.from_pretrained(config["decode_model"])
    tokenizer_decoder = AutoTokenizer.from_pretrained(config["decode_model"])
    criterion = SequenceCrossEntropyLoss()
    model_decoder.to(device)
    param_optimizer = list(model_decoder.named_parameters())
    no_decay = ["bias", "ln", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    num_gradients_accumulation = 1
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_train_optimization_steps = (
        len(dataloader) * num_epochs // num_gradients_accumulation
    )
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3, eps=1e-05)
    optimizer.add_param_group({"params": projection.parameters()})
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps
    )
    print("Model, optimizer and scheduler initialized")
    # =============================
    for i in range(num_epochs):
        model.eval()
        for idx, batch_data in enumerate(dataloader):
            prompts = [batch_data[i][0] for i in range(len(batch_data))]
            texts = [batch_data[i][1] for i in range(len(batch_data))]
            with torch.no_grad():
                embeddings = model.encode(texts, convert_to_tensor=True).to(device)
            embeddings = projection(embeddings)
            record_loss, perplexity = train_on_batch(
                batch_X=embeddings,
                batch_D=texts,
                batch_P=prompts,
                model=model_decoder,
                tokenizer=tokenizer_decoder,
                criterion=criterion,
                device=device,
                train=True,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()  # make sure no grad for GPT optimizer
            print(
                f"Training: epoch {i} batch {idx} / {len(dataloader)} with loss {record_loss} and PPL {perplexity}"
            )
        proj_path = f"projection_gpt2_{config['action_type']}.pt"
        torch.save(projection.state_dict(), proj_path)
        save_path = f"decoder_gpt2_{config['action_type']}"
        model_decoder.save_pretrained(save_path)
        print(f"Epoch {i} done")
    print("Training done")


def test(data, config):
    device = config["device"]
    batch_size = config["batch_size"]
    model = SentenceTransformer(config["embed_model"], device=device)  # dim 512
    save_path = f"decode_results_gpt2_{config['action_type']}.log"
    dataset = Corpus(data)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=dataset.collate,
    )
    print("Test data loaded")
    proj_path = f"projection_gpt2_{config['action_type']}.pt"
    projection = LinearProjection(in_dim=512, hidden_dim=256, out_dim=768)
    projection.load_state_dict(torch.load(proj_path))
    projection.to(device)
    print("Projection loaded")
    # setup on config for sentence generation   AutoModelForCausalLM
    decoder_path = f"decoder_gpt2_{config['action_type']}"
    config["model"] = AutoModelForCausalLM.from_pretrained(decoder_path).to(device)
    config["tokenizer"] = AutoTokenizer.from_pretrained(config["decode_model"])
    print("Decoder loaded")
    sent_dict = {}
    sent_dict["gt"] = []
    sent_dict["pred"] = []
    with torch.no_grad():
        for idx, batch_data in tqdm(enumerate(dataloader)):
            texts = [batch_data[i][1] for i in range(len(batch_data))]
            prompts = [batch_data[i][0] for i in range(len(batch_data))]
            embeddings = model.encode(texts, convert_to_tensor=True).to(device)
            embeddings = projection(embeddings)
            sent_list, gt_list = eval_on_batch(
                batch_X=embeddings,
                batch_D=texts,
                batch_P=prompts,
                model=config["model"],
                tokenizer=config["tokenizer"],
                device=device,
                config=config,
            )
            sent_dict["pred"].extend(sent_list)
            sent_dict["gt"].extend(gt_list)
        with open(save_path, "w") as f:
            json.dump(sent_dict, f, indent=4)
    print("Test done")


def train_on_batch(
    batch_X, batch_D, batch_P, model, tokenizer, criterion, device, train=True
):
    # batch_X: embedding
    # batch_D: text
    # batch_P: prompt, "" for post, original post text for repost and comment
    if not tokenizer.eos_token:
        tokenizer.eos_token = config["eos_token"]
    padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    batch_A = [f"{p}[SEP]{d}" for p, d in zip(batch_P, batch_D)]
    inputs = tokenizer(
        batch_A,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_ids = inputs["input_ids"].to(device)  # tensors of input ids
    labels = input_ids.clone()

    # embed the input ids using GPT-2 embedding
    input_emb = model.transformer.wte(input_ids)

    # add extra dim to cat together
    batch_X = batch_X.to(device)
    batch_X_unsqueeze = torch.unsqueeze(batch_X, 1)
    inputs_embeds = torch.cat(
        (batch_X_unsqueeze, input_emb), dim=1
    )  # [batch, max_length+1, emb_dim (1024)]
    past = None
    logits, past = model(
        inputs_embeds=inputs_embeds, past_key_values=past, return_dict=False
    )
    logits = logits[:, :-1].contiguous()
    target = labels.contiguous()
    target_mask = torch.ones_like(target).float()
    loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")
    record_loss = loss.item()
    perplexity = np.exp(record_loss)
    if train:
        loss.backward()
    return record_loss, perplexity


def eval_on_batch(batch_X, batch_D, batch_P, model, tokenizer, device, config):
    # batch_X: embedding
    # batch_D: text
    # batch_P: prompt, "" for post, original post text for repost and comment
    if not tokenizer.eos_token:
        tokenizer.eos_token = config["eos_token"]
    padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    inputs = tokenizer(
        [f"{p}[SEP]" for p in batch_P],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    input_ids = inputs["input_ids"].to(device)  # tensors of input ids
    input_masks = inputs["attention_mask"].to(device)
    valid_lengths = (
        torch.sum(input_masks, dim=1) - 1 + 1
    )  # -1 for [EOS], +1 for batch_X
    # embed the input ids using GPT-2 embedding
    input_emb = model.transformer.wte(input_ids)
    # add extra dim to cat together
    batch_X = batch_X.to(device)
    batch_X_unsqueeze = torch.unsqueeze(batch_X, 1)
    inputs_embeds = torch.cat((batch_X_unsqueeze, input_emb), dim=1)
    gt_list = batch_D
    sent_list = []
    for i, (inputs_embed, valid_length) in enumerate(zip(inputs_embeds, valid_lengths)):
        valid_inputs_embed = inputs_embed[
            :valid_length, :
        ]  # remove [EOS] and following invalid tokens
        # valid_inputs_embed: "X[CLS]P[SEP]" -> "TO-BE-DECODED...[EOS]"
        sentence = beam_decode_sentence(
            hidden_X=valid_inputs_embed, config=config, num_generate=1, beam_size=5
        )
        sentence = sentence[0]
        sent_list.append(sentence)
    return sent_list, gt_list


def main():
    parser = argparse.ArgumentParser(description="Training RECOSIM Decode Model")
    parser.add_argument(
        "--decode_model",
        type=str,
        help="The decode model you wanna fine-tune, such as gpt2",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        help="The encode model you use in the simulation",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="The path of the train data",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="The path of the test data",
    )
    parser.add_argument(
        "--action_type",
        type=str,
        help="The action type of the data",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="The number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="The device to run the model",
    )
    args = parser.parse_args()

    config = {}
    config["device"] = torch.device(args.device)
    config["decode_model"] = args.decode_model
    config["embed_model"] = args.embed_model
    config["tokenizer"] = AutoTokenizer.from_pretrained(args.decode_model)
    config["eos_token"] = config["tokenizer"].eos_token
    if not config["eos_token"]:
        config["eos_token"] = "[EOS]"

    config["num_epochs"] = args.num_epochs
    config["batch_size"] = args.batch_size
    config["action_type"] = args.action_type
    train_data = load_data(args.train_data_path)
    train(train_data, config)
    test_data = load_data(args.test_data_path)
    test(test_data, config)


if __name__ == "__main__":
    main()
