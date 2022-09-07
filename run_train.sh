#! /bin/sh

python train.py \
--do_train \
--load_best_model_at_end \
--overwrite_output_dir \
--description=SCD \
--eval_steps=2 \
--evaluation_strategy=steps \
--hidden_dropout_prob=0.05 \
--hidden_dropout_prob_noise=0.155 \
--alpha_unc=0.01 \
--lambda_unc=10.0 \
--learning_rate=3e-05 \
--max_seq_length=32 \
--max_steps=2 \
--metric_for_best_model=sickr_spearman \
--model_name_or_path=bert-base-uncased \
--n_projectors=2 \
--num_train_epochs=5 \
--output_dir=result \
--per_device_train_batch_size=96 \
--report_to=wandb \
--save_total_limit=0 \
--task_alpha=1 \
--task_beta=0.005225 \
--task_lambda=0.012 \
--temp=0.05 \
--train_file=data/wiki1m_for_simcse.txt \

