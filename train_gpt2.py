from argparse import ArgumentParser
import logging

from simpletransformers.language_modeling import (
        LanguageModelingModel,
        LanguageModelingArgs,
)

def train_gpt2():
    parser = ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("eval_path")
    parser.add_argument("outdir")
    parser.add_argument("--epochs", default=1, type=int)
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model_args = LanguageModelingArgs()
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.num_train_epochs = args.epochs
    model_args.dataset_type = "simple"
    model_args.mlm = False  # mlm must be False for CLM

    train_file = args.train_path
    test_file = args.eval_path

    model = LanguageModelingModel(
        "gpt2", "gpt2-medium", args=model_args
    )

    # Train the model
    model.train_model(train_file, eval_file=test_file, output_dir=args.outdir)

    # Evaluate the model
    result = model.eval_model(test_file)
    print(result)


if __name__ == "__main__":
    train_gpt2()