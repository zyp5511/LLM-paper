import json
import os
from argparse import ArgumentParser

WEBSITE_SUMMARIES = "website_summaries"
VERDICT = "verdict"
PROS = "pros"
CONS = "cons"
CUSTOMER_REVIEWS = "customer_reviews"
TEXT = "text"
VERIFIED = "verified"
HELPFUL = "helpful_votes"
PUBLISHED = "publication_date"
RATING = "rating"

PROMPT = "Act as an e-commerce expert. Summarize the following product reviews.\n"


def format_summaries():
    parser = ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("outdir")
    parser.add_argument("--output-type", choices=[VERDICT, PROS, CONS], default=VERDICT)
    args = parser.parse_args()

    for filename in os.listdir(args.indir):
        filepath = os.path.join(args.indir, filename)
        with open(filepath, 'r', encoding='utf8') as f:
            data = json.load(f)

        reviews = sorted(data[CUSTOMER_REVIEWS], key=lambda x: x[HELPFUL], reverse=True)
        negatives = [review for review in reviews if review[RATING] < 3.0]
        positives = [review for review in reviews if review[RATING] > 3.0]
        neutral = [review for review in reviews if review[RATING] == 3.0]

        if args.output_type == VERDICT:
            input_str = "\n".join([review[TEXT] for review in reviews[:4]])
            output = data[WEBSITE_SUMMARIES][0][args.output_type]
        elif args.output_type == PROS:
            input_str = "\n".join([review[TEXT] for review in positives[:4]])
            output = "\n".join(data[WEBSITE_SUMMARIES][0][args.output_type])
        elif args.output_type == CONS:
            input_str = "\n".join([review[TEXT] for review in negatives[:4]])
            output = "\n".join(data[WEBSITE_SUMMARIES][0][args.output_type])
        else:
            raise ValueError("No valid option for output type.")

        new_dict = {
            "input": PROMPT + input_str,
            "output": output
        }

        outpath = os.path.join(args.outdir, filename)
        with open(outpath, 'w', encoding='utf8') as outfile:
            json.dump(new_dict, outfile)

if __name__ == "__main__":
    format_summaries()