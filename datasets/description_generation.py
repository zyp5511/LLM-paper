import json
import os
from argparse import ArgumentParser
from typing import List, Dict

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
    parser.add_argument("amazon_meta_data")
    parser.add_argument("existing_dataset")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    existing_descripts = read_processed_data(args.existing_dataset)
    train = []
    dev_test = []
    for i, data in enumerate(yield_jsons(args.amazon_meta_data)):
        if i % 10_000 == 0:
            print(f"Completed {i}")
        if not data["description"]:
            continue
        description = "\n".join(data["description"])
        if description in existing_descripts:
            if len(train) < 2_000_000:
                train.append(
                    {
                        "input": PROMPT + data["title"],
                        "output": description,
                    }
                )
        else:
            if len(dev_test) < 50_000:
                dev_test.append(
                    {
                        "input": PROMPT + data["title"],
                        "output": description,
                    }
                )
        if len(dev_test) >= 50_000 and len(train) >= 2_000_000:
            break
    write_data(train, os.path.join(args.output_dir, "train.jsonl"))
    write_data(dev_test[:25000], os.path.join(args.output_dir, "dev.jsonl"))
    write_data(dev_test[25000:], os.path.join(args.output_dir, "test.jsonl"))


if __name__ == "__main__":
    description_generation()