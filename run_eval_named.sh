#! /bin/sh

FOLDER=$1
python scd_to_huggingface.py --path $FOLDER && \
python evaluation.py \
--pooler cls_before_pooler \
--task_set transfer \
--mode test \
--model_name_or_path $FOLDER \
--logfile logfile.json \
--testconfig heads1_noregu
