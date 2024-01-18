import os
import sys
folder = "./datasets/description_generation"
fns = os.listdir(folder)
fns = [fn for fn in fns if fn.endswith(".json")]
train_fns = [fn for fn in fns if fn.startswith("train")]
test_fn = [fn for fn in fns if fn.startswith("test")][0]
train_fns_full = [os.path.join(folder, fn) for fn in train_fns]
test_fn_full = os.path.join(folder, test_fn)

for fn in train_fns_full:
    n_sample = os.path.basename(fn).split("_")[1].split(".")[0]
    cmd = f"CUDA_VISIBLE_DEVICES=0 python ./summerization/run_summerirization.py \
    --model_name_or_path t5-base \
    --text_column input \
    --summary_column output \
    --train_file {fn} \
    --validation_file {test_fn_full} \
    --test_file {test_fn_full} \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --output_dir ./outputs/generation_{n_sample} \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --source_prefix ''\
    "
    # execute the command
    os.system(cmd)
    print(cmd)
    break