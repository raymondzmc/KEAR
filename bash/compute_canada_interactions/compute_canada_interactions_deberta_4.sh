#!/bin/bash
#SBATCH --gres=gpu:v100l:1      # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-0:00
#SBATCH --output=interactions_debertav3-large_4.out

module load python
export HF_HOME="/project/def-carenini/liraymo6"
source /home/liraymo6/virtualenvs/commonsense/bin/activate

export DATA_DIR="data/"
export OUTPUT_DIR="results/debertav3-large/"
export TOKENIZERS_PARALLELISM=false
export LOADMODEL_ERROR=0
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.4

deepspeed task.py --data_version csqa_ret_3datasets --append_descr 1 --append_retrieval 1 --append_triples --append_answer_text 1 \
               --preset_model_type debertav3-large --max_seq_length 512 --batch_size 1 --vary_segment_id  --seed 42 --local_rank 0 --mission output \
               --predict_dev --predict_dir results/debertav3-large/predictions --pred_file_name pred_test.csv  --output_model_dir results/debertav3-large \
                 --ddp --deepspeed --deepspeed_config debertav3-test --interpret_method interaction --num_partitions 10 --partition 4