import json
import os
import random
from argparse import ArgumentParser

random.seed(1)


def write_to_conll(filepath, sentences):
    with open(filepath, 'w', encoding='utf8') as out:
        for sent in sentences:
            for tup in sent:
                print(f"{tup[0]} {tup[1]}", file=out)
            print("", file=out)


def convert_open_brand():
    parser = ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outdir")
    args = parser.parse_args()

    sentences = []

    with open(args.infile, 'r', encoding='utf8') as f:
        for line in f:
            entry = json.loads(line)
            text = entry["description"]
            labels = entry["tag"]
            tokens = text.split()  # Seems already tokenized
            if len(labels) != len(tokens):
                print("Length mismatch")
                print(tokens, labels)
                continue
            sentences.append([(token, label) for token, label in zip(tokens, labels)])

    random.shuffle(sentences)
    train_index = int(len(sentences) * .7)
    dev_index = int(len(sentences) * .8)
    train = sentences[:train_index]
    dev = sentences[train_index:dev_index]
    test = sentences[dev_index:]

    write_to_conll(os.path.join(args.outdir, "train.txt"), train)
    write_to_conll(os.path.join(args.outdir, "dev.txt"), dev)
    write_to_conll(os.path.join(args.outdir, "test.txt"), test)


if __name__ == "__main__":
    convert_open_brand()