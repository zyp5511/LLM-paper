from typing import Dict, Optional, Sequence, List
import transformers
from transformers import AutoTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    ):

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_tokenizer_match_model(path, cache_dir, model_max_length, model, format):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        truncation_side="right",
        trust_remote_code=True,
        special_tokens_dict=None,
    )
    special_tokens_dict = {}
    if format == "oasst":
        special_tokens_dict = {
            "additional_special_tokens": ["<|prompter|>", "<|assistant|>"],
        }
    else:
        pass
    if not tokenizer.pad_token:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    if not tokenizer.pad_token:
        raise Exception("pad_token is not set in the pretrained model")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise Exception("pad_token_id and eos_token_id are the same")
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer
    