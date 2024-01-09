import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import NamedTuple, List
import pandas as pd

## Filenames
# train, dev, test.csv

ANSWER_GENERATION = "answer_generation"
# ASIN	question	candidate	answer	source

EVIDENCE_RANKING = "evidence_ranking"
# qid	ASIN	qa_pair_id	question	candidate	label	source	confidence

# TYPES_OF_EVIDENCE = {"attribute", "bullet", "desc", "cqa"}

FILENAMES = ["dev.csv", "test.csv", "train.csv"]

PROMPT = "Act as an e-commerce expert. Answer the following question based on the context. "

class QASample(NamedTuple):
    question: str
    answer: str
    evidence: List[str]

def format_semipqa():
    parser = ArgumentParser()
    parser.add_argument("hetpqa_dir")
    parser.add_argument("outdir")
    args = parser.parse_args()


    # Evidence Ranking
    ev_ranking = defaultdict(lambda: defaultdict(list))  # ASIN: Question: [candidate evidences]
    for filename in FILENAMES:

        filepath = os.path.join(args.hetpqa_dir, EVIDENCE_RANKING, filename)
        df_ev_rank = pd.read_csv(filepath, sep='\t', header=0)
        # print(filename, len(df_ev_rank))
        for index, row in df_ev_rank.iterrows():
            # if row["candidate"] in
            ev_ranking[row["ASIN"]][row["question"]].append(row["candidate"])

    samples = defaultdict(list)
    for filename in FILENAMES:  #
        filepath_answ = os.path.join(args.hetpqa_dir, ANSWER_GENERATION, filename)
        df_answer_gen = pd.read_csv(filepath_answ, sep='\t', header=0)
        # print(filename, len(df_answer_gen))
        for index, row in df_answer_gen.iterrows():

            candidates = ev_ranking[row["ASIN"]][row["question"]]
            if candidates:
                samples[filename].append(
                    QASample(
                        row["question"],
                        row["answer"],
                        candidates
                    )
                )

    for filename in FILENAMES:
        outpath = os.path.join(args.outdir, filename)
        with open(outpath, 'w', encoding='utf8') as outfile:
            for sample in samples[filename]:
                context = '\n'.join(sample.evidence)
                input_str = f"\nQuestion: {sample.question}\nContext: {context}"
                d = {
                    "input": PROMPT + input_str,
                    "output": sample.answer
        }
                print(json.dumps(d), file=outfile)




if __name__ == "__main__":
    format_semipqa()