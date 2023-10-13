import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import os
from os. path import exists, join, isdir
import json
import argparse
import torch
import transformers
from transformers import(
    set_seed,
    Seq2SeqTrainer,
    AutoTokenizer
)

from utils_tokenizer import get_tokenizer_match_model, smart_tokenizer_and_embedding_resize
from dataset import make_supervised_data_module
from model import get_accelerate_model, SavePeftModelCallback, safe_save_model_for_hf_trainer
import datetime
from transformers.integrations import is_tensorboard_available
torch.backends.cuda.matmul.allow_tf32 = True
import os
from torch import distributed as dist
from datetime import timedelta

# os.environ["http_proxy"] = "http://httpproxy-tcop.vip.ebay.com:80"
# os.environ["https_proxy"] = "http://httpproxy-tcop.vip.ebay.com:80"
# os.environ["WANDB _DISABLED"] = "true"
os.environ['CURL_CA_BUNDLE'] = ''
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="tiiuae/falcon-7b",
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    lora_all_modules: bool = field(
        default=True,
        metadata={"help":"Apply LoRA to all layers, qlora source code always this"},
    )

@dataclass
class DataArguments:
    dataset_names: str = field(
        default = "orca",
        metadata ={"help" : "Which dataset_name is used"}
    )
    train_data_fn: str = field(
        default = "/data/ebay-sic-a100/data/yipezhang/seqseq-refomulation-us-r2/train_complete.tsv",
        metadata={"help": "fn of the training data, it only support pandas supported fn"}
    )
    eval_data_fn: str = field(
        default="/data/ebay-slc-a100/data/yipezhang/seq2seq-refomulation-us-r2/test_complete.tsv",
        metadata={"help": "fn of the eval data, it only support pandas supported fn"}
    )
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"}
    )
    eval_subset: Optional[int] = field(
        default=50000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    stage: Optional[str] = field(
    default="sft",
    metadata={"help": "the stage of the training, SFT here"}
    )
    num_proc: int = field(
    default=48,
    metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str]= field(
        default = None
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). "}
        )
    #  training strategy: lora, qlora or full finetune
    finetune_type: str = field(
        default = "qlora",
        metadata={"help": "Finetune the entire model with adapters 'lora' or 'qlora' or without adapter 'full_finetune'."}
    )
    # bits used to load model. If full finetune model, bit need to be 16 or 32
    bits: int = field(
        default=4,
        metadata ={"help": "How many bits to use when loading model."}
    )
    # qlora setting
    adam8bit: bool = field(
        default=True,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization,"}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of 'fp4' or 'n$4'."}
    )
    # lora config ###
    lora_r: int = field(
        default=8,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )   
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Lora dropout."}
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Apply LoRA to all layers, qlora source code always this"},
    )
    lora_modules_to_save :Optional[List[str]] = field(
        default=None,
        metadata={"help": "Apply LoRA to all layers, qlora source code always this"},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Apply LoRA to all layers, qlora source code always this"},
    )

    max_memory_MB: int = field(
        default=24000,
        metadata={"help": "Free memory per gpu, here we use A100."}
    )
    output_dir: str = field(default='./Llama/sft', metadata={"help": 'The output dir for logs and checkpoints'})
    do_tratn: bool = field (default=True, metadata={"help": 'To train or not to train, that 1s the question?'})
    optim: str = field(default='paged_adamw_8bit', metadata={"help": 'The optimizer to be used' })
    # normal training hyperparameters
    pre_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'The number of gradient accumulation steps. Increase for better speed.'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    num_train_epochs: int = field(default=3, metadata={"help": 'How many epochs to take'})
    weight_decay: float = field(default=0.000, metadata={"help": 'The L2 weight decay rate of Adamw'}) # use Lora dropout instead for regularization if needed
    learning_rate: float = field(default=1e-5, metadata={"help": 'The learnign rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This 1s tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field (default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    lr_scheduler_type: str = field(default='cosine', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for!'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=200, metadata={"help": 'How often to save a model'})
    #save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'))
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'tensorboard' to log with tensorboard"})
    logging_dir: Optional[str] = field(default="./logs/sft", metadata={"help": "where to output the log"})
    eval_steps: int = field(default=200, metadata={"help": 'eval_step'})
    evaluation_strategy: str = field(default="steps", metadata={"help": 'evaluation_strategy'})
    logging_first_step: bool = field(default=True, metadata={"help": ' Logging_first_step'})
    torch_compile: bool = field(default=False, metadata={"help": 'Torch Compile'})
    seed: int = field(default=0, metadata={"help" :"seed"})

def train():
    # 
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    today = str(datetime.date.today())
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.dataset_names.replace("/", "-"), f'{training_args.finetune_type}_{today}_{training_args.learning_rate}')                         
    training_args.logging_dir = os.path.join(training_args.logging_dir,f'{data_args.dataset_names.replace("/", "-")}-{training_args.learning_rate}')
    args.dataset_names = args.dataset_names.split("#")
    set_seed(training_args.seed)

    ## here we can also use AutoTokenizer. from pretrained
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir = training_args.cache_dir,
        model_max_length=training_args.model_max_length,
    )

    # Normally we create dataset first for more time efficiently debugging
    data_module = make_supervised_data_module(tokenizer = tokenizer, args=args, max_length=training_args.model_max_length, ppo=False, use_sft= True)
    model, _ = get_accelerate_model(args, checkpoint_dir=None)
    model.config.use_cache = False # This is necessary because we need to
    print ('loaded model')

    special_tokens_dict = {}
    if not tokenizer.pad_token:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.pad_token_id is tokenizer.eos_token_id:
        raise Exception( 'pad_token_id should not be equal to eos_ token id')
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    #model = torch. compile (model)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )
    # Callbacks to save peft model
    if args.finetune_type == "lora" or args.finetune_type == "qlora":
        trainer.add_callback(SavePeftModelCallback)
    os.makedirs(training_args.output_dir, exist_ok = True)
    os.makedirs(training_args.logging_dir, exist_ok = True)

    # Save the updated tokenizer, including tokenizer.json, special_tokens_map.json, tokenizer_config.json
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "save_tokenizer"))
    all_metrics = {"run name": training_args.run_name}

    training_args.do_train = True
    # Training
    if training_args.do_train:
        logging.warning(f"*** Start training with {args.finetune_type} ** ")
        train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    all_metrics.update(metrics)
    if args.finetune.type =="full finetune": # full fine tuning will save the entire model
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if training_args.do_train:
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
    # save all arguments
    with open(os.path.join(training_args.output_dir, "params.txt"), "W") as fout:
        for key, value in vars(args).items():
            fout.write('%s: %s\n' % (key, value))


if __name__ == "__main__":
    train()