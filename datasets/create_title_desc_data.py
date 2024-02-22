import json
import os
from argparse import ArgumentParser
from typing import List, Dict

import random

from bs4 import BeautifulSoup
from tqdm import tqdm

random.seed(1)

PROMPT = "Act as an e-commerce expert. Given a product title, generate a description for the product.\nTitle: "
MAX_DESC_TOKENS = 300
MIN_DESC_TOKENS = 50


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


def remove_tags(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)

def description_generation():
    parser = ArgumentParser()
    parser.add_argument("amazon_meta_data")
    parser.add_argument("output_dir")
    parser.add_argument("--limit", default=6_000_000)
    args = parser.parse_args()

    dataset = []
    for i, example in enumerate(yield_jsons(args.amazon_meta_data)):
        title = example["title"]
        description_sents = example["description"]


        desc_tokens = " ".join(description_sents).split()
        title_tokens = title.split()

        if (
            not description_sents
            or len(desc_tokens) <= len(title_tokens)
            or len(desc_tokens) >= MAX_DESC_TOKENS
            or len(desc_tokens) <= MIN_DESC_TOKENS
        ):
            continue

        description_sents = [remove_tags(desc) for desc in description_sents]
        description_cleaned = " ".join(description_sents)

        dataset.append(
            {
                "input": title,
                "output": description_cleaned,
            }
        )
        if i % 10000 == 0:
            print(f"Completed {i} examples")
        # Check if we have probably enough data
        if i > args.limit and len(dataset) >= 600_000:
            break

    data_1k = random.sample(dataset, 1000)
    data_5k = random.sample(dataset, 5000)
    data_10k = random.sample(dataset, 10_000)
    data_50k = random.sample(dataset, 50_000)
    data_100k = random.sample(dataset, 100_000)
    data_150k = random.sample(dataset, 150_000)
    data_200k = random.sample(dataset, 200_000)

    res_data = []
    all_samples = data_1k + data_5k + data_10k + data_50k + data_100k + data_150k + data_200k
    used_set = set([d["output"] for d in all_samples])
    for datapoint in tqdm(dataset):
        if datapoint["output"] not in used_set:
            res_data.append(datapoint)
    test = random.sample(res_data, 10_000)

    res_data2 = []
    all_samples = data_1k + data_5k + data_10k + data_50k + data_100k + data_150k + data_200k + test
    used_set = set([d["output"] for d in all_samples])
    for datapoint in tqdm(dataset):
        if datapoint["output"] not in used_set:
            res_data2.append(datapoint)
    dev = random.sample(res_data2, 10_000)

    # used_set = used_set.union(set([d["output"] for d in test]))
    #
    # res_data2 = []
    # for datapoint in tqdm(dataset):
    #     if datapoint["output"] not in used_set:
    #         res_data2.append(datapoint)
    # train_150k = random.sample(res_data, 150_000)
    # train_200k = data_50k + train_150k

    os.makedirs(args.output_dir, exist_ok=True)

    write_data(data_1k, os.path.join(args.output_dir, "train_1k.jsonl"))
    write_data(data_5k, os.path.join(args.output_dir, "train_5k.jsonl"))
    write_data(data_10k, os.path.join(args.output_dir, "train_10k.jsonl"))
    write_data(data_50k, os.path.join(args.output_dir, "train_50k.jsonl"))
    write_data(data_100k, os.path.join(args.output_dir, "train_100k.jsonl"))
    write_data(data_150k, os.path.join(args.output_dir, "train_150k.jsonl"))
    write_data(data_200k, os.path.join(args.output_dir, "train_200k.jsonl"))
    write_data(dev, os.path.join(args.output_dir, "dev.jsonl"))
    write_data(test, os.path.join(args.output_dir, "test.jsonl"))


if __name__ == "__main__":
    description_generation()