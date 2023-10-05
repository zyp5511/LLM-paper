import os
from os.path import exists, join, isdir
import logging

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import bitsandbytes as bnb
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConf1g,
    LlamaTokenizer,
    AutoModelForSequenceClassification
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers. trainer_utils import PREFIX_CHECKPOINT_DIR
#from tr import AutoModelForCausalLMWithValueHead
torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger(__name__)

####lora all linear modules ####
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bit if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names: # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


#### Save PEFT model checkpoint ####
class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print(' Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"(PREFIX_CHECKPOINT _DIR)-(state.global_step)")
        peft_model_path = os. path. join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        pytorch_model_path=os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
    def on_save(self, args, state, control, kwargs):
        if not control.should_save:
            return
        self.save_model(args, state, kwargs)
        return control
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return super().on_train_begin(args, state, control, **kwargs)
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        def touch(path, times=None):
            with open(path, "a"):
                os.utime(path, times)

        touch(os.path.join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)
    
def safe_save_model_for_hf_trainer(trainer:transformers.Trainer,output_dir:str):
    """Collects the state dict of the model and saves it to the output_dir"""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save_state(output_dir, state_dict=cpu_state_dict)

def load_model(args, device_map, max_memory, compute_dtype):
    if args.stage.lower()=='rw':
        if args.finetune_type == "lora" or args.f1netune_type == "full_finetune":
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                device_map=device_map,
                trust_remote_code=True,
                num_labels=1,
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                torch_dtypes=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            )
        elif args.finetune_type == "qlora":
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                num_labels=1,
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                device_map=device_map,
                max_memory=max_memory,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=args.bits == 4,
                    load_in_8bit=args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=args.double_quant,
                    bnb_4bit_quant_type=args.quant_type,
                    torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                ),
                trust_remote_code=True,
            )
    elif args.stage.lower() == "sft" or args.stage.lower() == "ppo":
        if args.finetune_type == "lora" or args.finetune_type == "full_finetune":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                trust_remote_code=True,
            )
        elif args.finetune_type == "qlora":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                device_map=device_map,
                max_memory=max_memory,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=args.bits == 4,
                    load_in_8bit=args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=args.double_quant,
                    bnb_4bit_quant_type=args.quant_type,
                    torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                ),
                trust_remote_code=True,
            )
        else:
            raise NotImplementedError("finetune type not implemented")
    model.config.torch_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    return model

def get_accelerate_model(args, checkpoint_dir):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}

    device_map = "auto"
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {0: local_rank}
        max_memory = {local_rank: f'{args.max_memory_MB}MB'}
    if args.finetune_type == "full_finetune":  
        assert args.bits in (16, 32)
    print(f'Loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = load_model(args, device_map, max_memory, compute_dtype)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    if args.finetune_type == "lora" or args.finetune_type == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.finetune_type == "lora" or args.finetune_type == "qlora":
        if checkpoint_dir is not None:
            print("checkpoint_dir is NOT None. Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, "adapter_model"), is_trainable=True)
        else:
            print(f'checkpoint_dir is None. Adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                re=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                modules_to_save=args.lora_modules_to_save,
                task_type=args.lora_task_type
            )
            model = get_peft_model(model, config)
        if compute_dtype == torch.float16 and args.bits == 4:
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:
                print('='*80)
                print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
                print('='*80)
            else:
                if args.bf16:
                    raise ValueError('Your GPU does not support bfloat16')
    
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(dtype=torch.bfloat16)
        if 'norm' in name:
            module = module.to(dtype=torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module.weight = module.weight.to(dtype=torch.bfloat16)
    return model, device_map
