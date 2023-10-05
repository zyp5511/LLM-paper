import copy, os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import json
import torch
import transformers
from datasets import load_dataset, Dataset, DatasetDict, DownloadConfig, concatenate_datasets
import numpy as np
from fastchat.conversation import Conversation, get_conv_template
IGNORE_INDEX = -100

@dataclass
class DataCollatorForSupervisedDataset(object):
    #"Collate examples for supervised fine-tuning."
    tokenizer:transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        if "labels" in instances[0]:
            input_ids, labels = tuple([instance[key] for instance in instances]for key in ("input_ids", "labels") )
            input_ids= torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        else:
            input_ids = [instance["input ids"] for instance in instances]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value = self.tokenizer.pad_token_id)
            labels = input_ids
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def get_orca(dataset_name, num_proc, seed = 0, download_conf= None, ptx = False):
    dataset_name = "Open-Orca/OpenOrca"
    base_ur = "https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/"
    data_files = {"train": base_ur + "1M-GPT4-Augmented. parquet"}
    dataset = load_dataset("parquet", data_f1les=data_files, split="train",
                            download_conf=download_conf).select(np.arange(50000)) 
    dataset = dataset.rename_column ("question", "prompt")
    dataset_split = dataset.train_test_split(test_size=0.1)
    train_data = dataset_split["train"]
    valid_data = dataset_split("test")
    return train_data, valid_data

def get_ebay_data(dataset_name, num_proc, seed = 0, download_conf= None, ptx = False):
    pass

def create_prompt_template(prompt, response = None, system_message = None) :
    # This can be done in a very simple way:
    # Template:
    # ‹SYS> system_message «\SYS> «INS» prompt «\INS> response «ls>
    # if system_message is None,
    # eSYS> «\SYS> «INS> prompt «\INS> response <\s>
    # Or you can manually make it at
    # «INS> prompt «\INS> response e\s>
    # If response is None
    # «SYS> elSYS> «INS> prompt «\INS> e\s>

    conv = get_conv_template("llama-2")
    if system_message is not None:
        conv.set_system_message(system_message)
    conv.set_system_message(" ")
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], response)
    res = conv.get_prompt()
    return res

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, args, use_sft = True, ppo = False, max_length=0, select_train= -1, select_eval=-1, ptx=False):
    '''''
    Make dataset and collator for supervised fine-tuning.
    if use_sft = True, dataset keys = input_ids, labels, Input_ids is corresponding to query + response, label is masked by the query
    if ppo = True, dataset keys = Input_ids, input_ids 1s corresponding to query , label is none
    '''''

    def preprocess(examples):
        """Preprocess the data by tokenizing."""
        res = {
            "Input, ids": [],
            "labels": [],
            }
        for query, response in zip(examples["prompt"], examples['response']):
            if query is None or response is None:
                continue
            query, response = query.strip(), response.strip()
            prompt = create_prompt_template(query, None)
            query_ids = tokenizer (prompt, return_tensors="pt")["input_ids"][0]
            if not ppo:
                prompt = create_prompt_template(query, response)
            tokenized = tokenizer(prompt, truncation=True, return_tensors="pt")
            input_ids = tokenized["input_ids"][0]
            labels = input_ids.clone()
            if use_sft:
                labels [:len(query_ids)] = IGNORE_INDEX # ignore the source, only calcaute loss on target
            res["input_ids"].append(input_ids)
            res ["labels"]. append (labels)
        return res
    if max_length == 0:
        max_length = tokenizer.model_max_length
    num_proc = 48
    train_list, eval_list = [], []
    '''
     We first make sure all the dataset have the same column "prompt", "response", then tokenize them. based on the conversation
    '''
    if type(args.dataset_names) is not list:
        args.dataset_names = args.dataset_names.split("#")
    for dataset_name in args.dataset_names:
        if "orca" in dataset_name:
            get_dateset_function = get_orca
        elif "ebay" in dataset_name:
            get_dateset_function = get_ebay_data
        else:
            raise NotImplementedError("dataset not implemented")
        train_data, valid_data = get_dateset_function(dataset_name, num_proc, args.seed, download_config=None, ptx=ptx)
        if select_train != -1:
            train_data = train_data.select(np.arange(min(select_train, len(train_data))).astype(int))
        if select_eval != -1:
            valid_data = valid_data.select(np.arange(min(select_eval, len(valid_data))).astype(int))
        print(f"{dataset_name}, Size of the train set: (len(train_data)). Size of the validation set: (len(valid_data)]")
        train_dataset = train_data.map(preprocess, batched=True, num_proc=num_proc).with_format("torch")
        eval_dataset = valid_data.map(preprocess, batched=True, num_proc=num_proc).with_format("torch")
        train_dataset= train_dataset.f1lter (lambda x: len(x["input_ids"]) < max_length)
        eval_dataset = eval_dataset.f1lter (lambda x: len(x[" input_1ds"]) < max_length)
        train_list.append(train_dataset)
        eval_list.append(eval_dataset)
    train_dataset, eval_dataset = concatenate_datasets(train_list).with_format("torch"), concatenate_datasets(eval_list).with_format ("torch")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


