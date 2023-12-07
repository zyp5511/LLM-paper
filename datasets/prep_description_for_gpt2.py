import json
import re
from argparse import ArgumentParser


def prep_description_data():
    parser = ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()


    with open(args.input_path, 'r', encoding='utf8') as infile, open(args.output_path, 'w', encoding='utf8') as outfile:
        for line in infile:
            data = json.loads(line)
            input_str = re.sub(r'\n', '', data['input'])
            output_str = re.sub(r'\n', '', data['output'])
            print(f"{input_str} <SEP> {output_str}",
                file=outfile
            )


if __name__ == "__main__":
    prep_description_data()