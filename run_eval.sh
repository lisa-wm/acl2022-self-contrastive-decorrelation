#! /bin/sh

LATESTRESULT=$(ls -td result/* | head -1)
python scd_to_huggingface.py --path LATESTRESULT && \
python evaluation.py \
--pooler cls_before_pooler \
--task_set transfer \
--mode test \
--model_name_or_path LATESTRESULT \
--logfile logfile.json

