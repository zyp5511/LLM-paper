import json
import re
from argparse import ArgumentParser

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


TUPLE_PATTERN = re.compile(r"(\w+?,\s+[BI]-\w+|\w+?,\s+O)")
OUTPUT = "output"
MODEL_OUTPUT = "model_output"


def get_ner_tuples(tuple_strings):
    # print(tuple_strings)
    labels = []
    tokens = []
    for tup in tuple_strings:
        pieces = tup.split(',')
        token = pieces[0].strip()
        label = pieces[1].strip()
        labels.append(label)
        tokens.append(token)
    return tokens, labels

def convert_conll(example):
    true_output = example[OUTPUT]
    model_output_str = example[MODEL_OUTPUT]

    tuple_strings = TUPLE_PATTERN.findall(true_output)
    true_tokens, true_labels = get_ner_tuples(tuple_strings)
    model_tuple_strings = TUPLE_PATTERN.findall(model_output_str)
    pred_tokens, pred_labels = get_ner_tuples(model_tuple_strings)

    # If there's mismatch in length
    # Assume model messed up and make predicted labels all Os
    # This may not be ideal, consider giving partial credit in future? But it'd be a bit dicey
    if len(true_labels) != len(pred_labels):
        pred_labels = ["O" for _ in range(len(true_labels))]

    return true_labels, pred_labels


def convert_to_bio():
    parser = ArgumentParser()
    parser.add_argument("input", help="Json lines output with true output labels and model predictions.")
    args = parser.parse_args()

    true_labels = []
    pred_labels = []
    
    with open(args.input, 'r', encoding='utf8') as f:
        for line in f:
            example = json.loads(line)
            trues, preds = convert_conll(example)
            true_labels.append(trues)
            pred_labels.append(preds)

    print(classification_report(true_labels, pred_labels, scheme=IOB2))


if __name__ == "__main__":
    convert_to_bio()