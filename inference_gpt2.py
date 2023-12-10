import json
import re
from argparse import ArgumentParser

from simpletransformers.language_generation import LanguageGenerationModel
from tqdm import tqdm

OUTPUT_PATTERN = re.compile(r'\"output\":(.*)')


def run_inference():

    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--evalpath")
    parser.add_argument("--outpath")
    args = parser.parse_args()

    model = LanguageGenerationModel("gpt2", args.model)

    with open(args.evalpath, 'r', encoding='utf8') as infile, open(args.outpath, 'w', encoding='utf8') as out:
        for line in tqdm(infile):
            example = json.loads(line)
            sys_out = model.generate(prompt=example['input'], args={"max_length": 128})
            match = OUTPUT_PATTERN.search(sys_out[0])
            example['model_output'] = match.group(1)
            print(json.dumps(example), file=out)


if __name__ == "__main__":
    run_inference()