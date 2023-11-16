import copy, os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import json
import torch
import transformers
from datasets import load_dataset, Dataset, DatasetDict, DownloadConfig, concatenate_datasets
import numpy as np
IGNORE_INDEX = -100

@dataclass
class DataCollatorForSupervisedDataset(object):
    #"Collate examples for supervised fine-tuning."
    tokenizer:transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids= torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def get_orca(dataset_name, num_proc, seed=0, download_config=None, ptx=False):
    dataset_name = "Open-Orca/OpenOrca"
    # base_ur = "https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/"
    base_ur = "./dataset/OpenOrca/"
    data_files = {"train": base_ur + "1M-GPT4-Augmented.parquet"}
    dataset = load_dataset("parquet", data_files=data_files, split="train",
                            download_config=download_config).select(np.arange(50000)) 
    dataset = dataset.rename_column("question", "prompt")
    dataset_split = dataset.train_test_split(test_size=0.001)
    train_data = dataset_split["train"]
    valid_data = dataset_split["test"]
    return train_data, valid_data



def get_esci_data(dataset_name, num_proc, seed=0, download_config=None, ptx=False):
    if dataset_name == "esci_task2_1k":
        data_path = "./ecal_colla/esci_task2_train_prompt_1k.json"
    elif dataset_name == "esci_task2_10k":
        data_path = "./ecal_colla/esci_task2_train_prompt_10k.json"
    elif dataset_name == "esci_task2_100k":
        data_path = "./ecal_colla/esci_task2_train_prompt_100k.json"
    elif dataset_name == "esci_task2_1m":
        data_path = "./ecal_colla/esci_task2_train_prompt_1m.json"
        
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.rename_column("input", "prompt")
    dataset = dataset.rename_column("output", "response")
    dataset_split = dataset.train_test_split(test_size=0.001)
    train_data = dataset_split["train"]
    valid_data = dataset_split["test"]
    return train_data, valid_data


def generate_prompt(instruction, input=None):
    PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is not None:
        return PROMPT_DICT["prompt_input"].format_map(dict(instruction=instruction, input=input))
    else:
        return PROMPT_DICT["prompt_no_input"].format_map(dict(instruction=instruction))
    
    

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, args, use_sft=True, ppo=False, max_length=0, select_train=-1, select_eval=-1, ptx=False):
    '''''
    Make dataset and collator for supervised fine-tuning.
    if use_sft = True, dataset keys = input_ids, labels, Input_ids is corresponding to query + response, label is masked by the query
    if ppo = True, dataset keys = Input_ids, input_ids 1s corresponding to query , label is none
    '''''

    def preprocess(examples):
        """Preprocess the data by tokenizing."""
        res = {
            "input_ids": [],
            "labels": [],
            }
        for query, response in zip(examples["prompt"], examples['response']):
            if query is None or response is None:
                continue
            query, response = query.strip(), response.strip()
            
            prompt = generate_prompt(query, None)
            query_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            
            prompt = generate_prompt(query) + response + tokenizer.eos_token ### very important here!!!
            
            tokenized = tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"][0]
            labels = input_ids.clone()
            labels[:len(query_ids)] = IGNORE_INDEX # ignore the source, only calcaute loss on target
            
            res["input_ids"].append(input_ids)
            res["labels"].append(labels)
        
        return res
    
    if max_length == 0:
        max_length = tokenizer.model_max_length
        
    num_proc = args.num_proc

    train_list, eval_list = [], []
    '''
     We first make sure all the dataset have the same column "prompt", "response", then tokenize them. based on the conversation
    '''
    if type(args.dataset_names) is not list:
        args.dataset_names = args.dataset_names.split("#")
    
    for dataset_name in args.dataset_names:
        if "orca" in dataset_name:
            get_dateset_function = get_orca
        elif "esci_task2" in dataset_name:
            get_dateset_function = get_esci_data
        else:
            raise NotImplementedError("dataset not implemented")
        
        train_data, valid_data = get_dateset_function(dataset_name, num_proc, args.seed, download_config=None, ptx=ptx)
        
        if select_train != -1:
            train_data = train_data.select(np.arange(min(select_train, len(train_data))).astype(int))
        if select_eval != -1:
            valid_data = valid_data.select(np.arange(min(select_eval, len(valid_data))).astype(int))
        print(f"{dataset_name}, Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
        
        train_dataset = train_data.map(preprocess, batched=True, num_proc=num_proc).with_format("torch")
        eval_dataset = valid_data.map(preprocess, batched=True, num_proc=num_proc).with_format("torch")
        train_dataset= train_dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
        eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
        train_list.append(train_dataset)
        eval_list.append(eval_dataset)
    
    train_dataset, eval_dataset = concatenate_datasets(train_list).with_format("torch"), concatenate_datasets(eval_list).with_format("torch")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    import random
    for index in random.sample(range(len(train_dataset)), 2):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")
        print(f"Sample {index} input: {tokenizer.decode(train_dataset[index]['input_ids'])}")
        
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)