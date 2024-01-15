import os
import sys
cmd = "python ./summerization/run_summerirization.py \
    --model_name_or_path t5-base \
    --text_column inputs \
    --summary_column targets \
    --train_file ./datasets/AmaSum/verdict/train.csv \
    --validation_file ./datasets/AmaSum/verdict/valid.csv \
    --test_file ./datasets/AmaSum/verdict/test.csv \
    --output_dir ./outputs/verdict \
    --do_train \
    --source_prefix "summarize: " \
    "
os.system(cmd)
