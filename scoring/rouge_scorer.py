import json
from argparse import ArgumentParser

import evaluate as evaluate


def score_rouge():
    parser = ArgumentParser()
    parser.add_argument("predictions")
    args = parser.parse_args()

    rouge = evaluate.load('rouge')

    predictions = []
    references = []
    with open(args.predictions, 'r', encoding='utf8') as f:
        for line in f:
            example = json.loads(line)
            predictions.append(example["model_output"])
            references.append(example["output"])
    results = rouge.compute(predictions=predictions, references=references)
    print(results)


if __name__ == "__main__":
    score_rouge()