import json
import os
from argparse import ArgumentParser
from typing import List, Dict

import random

from tqdm import tqdm

random.seed(1)

PROMPT = "Act as an e-commerce expert. Given a product title, generate a description for the product.\nTitle: "


def yield_jsons(filepath):
    with open(filepath, 'r', encoding='utf8') as infile:
        for line in infile:
            yield json.loads(line)


def read_processed_data(existing_dataset):
    descrs = set()
    for data in yield_jsons(existing_dataset):
            descrs.add(data["output"])
    return descrs


def write_data(examples: List[Dict], outpath: str) -> None:
    with open(outpath, 'w', encoding='utf8') as out:
        for example in examples:
            print(json.dumps(example), file=out)


def description_generation():
    parser = ArgumentParser()
    # parser.add_argument("amazon_meta_data")
    parser.add_argument("existing_dataset")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    dataset = list(yield_jsons(args.existing_dataset))


    data_1k = random.sample(dataset, 1000)
    data_5k = random.sample(dataset, 5000)
    data_10k = random.sample(dataset, 10_000)
    data_50k = random.sample(dataset, 50_000)

    res_data = []
    all_samples = data_1k + data_5k + data_10k + data_50k
    used_set = set([d["output"] for d in all_samples])
    for datapoint in tqdm(dataset):
        if datapoint["output"] not in used_set:
            res_data.append(datapoint)
    test = random.sample(res_data, 10_000)

    used_set = used_set.union(set([d["output"] for d in test]))

    res_data2 = []
    for datapoint in tqdm(dataset):
        if datapoint["output"] not in used_set:
            res_data2.append(datapoint)
    train_150k = random.sample(res_data, 150_000)
    train_200k = data_50k + train_150k

    write_data(data_1k, os.path.join(args.output_dir, "train_1k.jsonl"))
    write_data(data_5k, os.path.join(args.output_dir, "train_5k.jsonl"))
    write_data(data_10k, os.path.join(args.output_dir, "train_10k.jsonl"))
    write_data(data_50k, os.path.join(args.output_dir, "train_50k.jsonl"))
    write_data(train_200k, os.path.join(args.output_dir, "train_200k.jsonl"))
    # write_data(dev, os.path.join(args.output_dir, "dev.jsonl"))
    write_data(test, os.path.join(args.output_dir, "test.jsonl"))


    # existing_descripts = read_processed_data(args.existing_dataset)
    # train = []
    # dev_test = []
    # for i, data in enumerate(yield_jsons(args.amazon_meta_data)):
    #     if i % 10_000 == 0:
    #         print(f"Completed {i}")
    #     if not data["description"]:
    #         continue
    #     description = "\n".join(data["description"])
    #     if description in existing_descripts:
    #         if len(train) < 2_000_000:
    #             train.append(
    #                 {
    #                     "input": PROMPT + data["title"],
    #                     "output": description,
    #                 }
    #             )
    #     else:
    #         if len(dev_test) < 50_000:
    #             dev_test.append(
    #                 {
    #                     "input": PROMPT + data["title"],
    #                     "output": description,
    #                 }
    #             )
    #     if len(dev_test) >= 50_000 and len(train) >= 2_000_000:
    #         break
    # write_data(train, os.path.join(args.output_dir, "train.jsonl"))
    # write_data(dev_test[:25000], os.path.join(args.output_dir, "dev.jsonl"))
    # write_data(dev_test[25000:], os.path.join(args.output_dir, "test.jsonl"))


if __name__ == "__main__":
    description_generation()