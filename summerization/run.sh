CUDA_VISIBLE_DEVICES=0 python ./summerization/summerirization.py \
    --model_name_or_path t5-base \
    --text_column inputs \
    --summary_column target \
    --train_file ./datasets/AmaSum/verdict/train.csv \
    --validation_file ./datasets/AmaSum/verdict/valid.csv \
    --test_file ./datasets/AmaSum/verdict/test.csv \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --output_dir ./outputs/verdict \
    //--do_train \
    --do_predict \
    --predict_with_generate \
    --source_prefix "summarize: "