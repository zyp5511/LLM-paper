import json
import os
import random
from argparse import ArgumentParser
from typing import List, Dict

from tqdm import tqdm

random.seed(1)

def yield_jsons_from_dir(dirpath):
    print(f"{dirpath=}")
    for filename in os.listdir(dirpath):
        with open(os.path.join(dirpath, filename), 'r', encoding='utf8') as infile:
            for line in infile:
                yield json.loads(line)


def write_data(examples: List[Dict], outpath: str) -> None:
    with open(outpath, 'w', encoding='utf8') as out:
        for example in examples:
            print(json.dumps(example), file=out)

def description_generation():
    parser = ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_data = list(yield_jsons_from_dir(os.path.join(args.input_dir, "train")))
    dev_data = list(yield_jsons_from_dir(os.path.join(args.input_dir, "valid")))
    test_data = list(yield_jsons_from_dir(os.path.join(args.input_dir, "test")))

    train_data_1k = random.sample(train_data, 1000)
    train_data_5k = random.sample(train_data, 5000)
    train_data_10k = random.sample(train_data, 10000)
    train_data_25k = random.sample(train_data, 25000)

    write_data(train_data_1k, os.path.join(args.output_dir, "train_1k.jsonl"))
    write_data(train_data_5k, os.path.join(args.output_dir, "train_5k.jsonl"))
    write_data(train_data_10k, os.path.join(args.output_dir, "train_10k.jsonl"))
    write_data(train_data_25k, os.path.join(args.output_dir, "train_25k.jsonl"))
    write_data(test_data, os.path.join(args.output_dir, "test.jsonl"))
    write_data(dev_data, os.path.join(args.output_dir, "dev.jsonl"))
    write_data(train_data, os.path.join(args.output_dir, "train.jsonl"))


if __name__ == "__main__":
    description_generation()