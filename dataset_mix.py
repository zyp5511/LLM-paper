import copy, os
from typing import Dict
import transformers
from dataset import DataCollatorForSupervisedDataset, create_prompt_template
import numpy as np

import pandas as pd
from datasets import Dataset
IGNORE_INDEX = -100
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, args, use_sft = True, ppo = False, max_length=0, select_train= -1, select_eval=-1, only_sft_data = False):
    # Make dataset and collator for supervised fine-tuning.
    # if use_sft = True, dataset keys = input_ids, labels, input_ids is corresponding to query + response, label is masked by the query
    # if ppo = True, dataset keys = input_ids, input_ids is corresponding to query , label is none
    def preprocess(examples):
        '''
        Preprocess the data by tokenizing.
        Column requirements, 4 cols
        pretrain_marker : int 1 or 0
        system_message : str
        prompt : str
        response : str
        if pretrain_marker == 1:
            the data is constructed by "unsupervised learning"
        else:
            the data is constructed by "supervised fine turning"
        '''
        res = {
            "input, ids": [],
            "labels": [],
        }
        for pretrain_marker, sys, query, response in zip(examples ["pretrain_marker"], examples["system_message"], examples["prompt"], examples['response']):
            if pretrain_marker is None or query is None: ## two most important
                continue
            if pretrain_marker:
                prompt = query.strip() + tokenizer.eos_token
            else:
                if sys is None or response is None:
                    continue
            query, response = query.strip(), response.strip()
            prompt = create_prompt_template(query, None, system_message=sys)
            query_ids = tokenizer(prompt, return_tensors="pt")["input ids"][0]
            if not ppo:
                prompt = create_prompt_template(query, response, system_message=sys)
            tokenized = tokenizer(prompt, truncation=True, return_tensors="pt")
            input_ids = tokenized["input_ids"][0]
            labels = input_ids.clone()
            if use_sft and not pretrain_marker:
                labels [:len(query_ids)] = IGNORE_INDEX
            res["input_ids"].append(input_ids)
            res["labels"].append(labels)
        return res
    if max_length == 0:
        max_length = tokenizer.model_max_length
    num_proc = 48
    train_data, valid_data = pd.read_csv(args.train_data_fn), pd.read_csv(args.eval_data_fn)
    if only_sft_data:
        train_data = train_data[train_data["pretrain_marker"] == 0]
        valid_data = valid_data[valid_data["pretrain_marker"] == 0]
    train_data = Dataset.from_pandas(train_data)
    valid_data = Dataset.from_pandas(valid_data)
    if select_train != -1:
        train_data = train_data.select(np.arange(min(select_train, len(train_data))).astype(int))
    if select_eval != -1:
        valid_data = valid_data.select(np.arange(min(select_eval, len(valid_data))).astype(int))
    print("Size of the train set: ", len(train_data), ". Size of the validation set: ", len(valid_data))
    train_data = train_data.map(preprocess, batched=True, remove_columns=["pretrain_marker", "system_message", "prompt", "response"])
    valid_data = valid_data.map(preprocess, batched=True, remove_columns=["pretrain_marker", "system_message", "prompt", "response"])
    train_data.set_format("torch", columns=["input_ids", "labels"])
    valid_data.set_format("torch", columns=["input_ids", "labels"])
    collator = DataCollatorForSupervisedDataset(tokenizer, max_length=max_length, padding=True)
    return dict(
        train_dataset=train_data,
        valid_dataset=valid_data,
        data_collator=collator,
    )

