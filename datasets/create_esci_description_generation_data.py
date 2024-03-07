"""
Script to process ESCI data into desription generation task.
https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset
"""
import json
import os
from argparse import ArgumentParser
from functools import partial

from bs4 import BeautifulSoup
from datasets import load_dataset


def remove_tags(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


def filter_func(x, seen_titles):
    desc = x["product_description"]
    title = x["product_title"]
    bullet = x["product_bullet_point"]
    condition = (
        title is not None
        and desc is not None
        and bullet is not None
        and desc != title
        and desc != bullet
        and title not in seen_titles
        and len(title.split()) < len(desc.split())
    )
    # figure out if true before adding title to seen titles so don't skip every title
    seen_titles.add(title)
    return condition

def make_example(example):
    desc = remove_tags(example["product_description"])
    title = remove_tags(example["product_title"])
    bullet = remove_tags(example["product_bullet_point"])

    brand = example["product_brand"]
    color = example["product_color"]
    product_info = f"Title: {title}\nBrand: {brand}\nColor: {color}\nBullet points: {bullet}"
    example["input"] = product_info
    example["output"] = desc
    return example


def create_description_generation_from_esci():
    parser = ArgumentParser()
    parser.add_argument("outdir")
    parser.add_argument("--num-processors", default=1, type=int)
    args = parser.parse_args()

    seen_titles = set()

    dataset = load_dataset("tasksource/esci")
    en_data = dataset.filter(lambda x: x["product_locale"] == "us" and x["product_description"])
    print(en_data)
    filter_partial = partial(filter_func, seen_titles=seen_titles)
    en_data = en_data.filter(filter_partial)
    print(en_data)
    cols = en_data['train'].column_names

    os.makedirs(args.outdir, exist_ok=True)
    mapped_data = en_data.map(make_example, num_proc=args.num_processors)
    mapped_data = mapped_data.remove_columns(cols)

    print("Writing to files...")
    for split in mapped_data:
        outpath = os.path.join(args.outdir, f'{split}.jsonl')
        with open(outpath, 'w', encoding='utf8') as out:
            for example in mapped_data[split]:
                print(json.dumps(example), file=out)


if __name__ == "__main__":
    create_description_generation_from_esci()