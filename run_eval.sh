#! /bin/sh

python scd_to_huggingface.py --path result/hardy-lion-21 && \
python evaluation.py \
--pooler cls_before_pooler \
--task_set transfer \
--mode test \
--model_name_or_path result/hardy-lion-21 \
--logfile logfile.json

