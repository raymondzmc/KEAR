#!/bin/bash
#SBATCH --gres=gpu:v100l:2      # Request GPU "generic resources"
#SBATCH --cpus-per-task=12  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=2-0:00
#SBATCH --output=interpret_debertav3-large.out

module load python
export HF_HOME="/project/def-carenini/liraymo6"
source /home/liraymo6/virtualenvs/commonsense/bin/activate

export DATA_DIR="data/"
export OUTPUT_DIR="results/debertav3-large/"
export TOKENIZERS_PARALLELISM=false
export LOADMODEL_ERROR=0
export CUDA_VISIBLE_DEVICES=0,1

deepspeed task.py --data_version csqa_ret_3datasets --append_descr 1 --append_retrieval 1 --append_triples --append_answer_text 1 \
               --preset_model_type debertav3-large  --max_seq_length 512 --batch_size 1 --vary_segment_id  --seed 42 --local_rank 0 \
               --predict_dir results/debertav3-large/predictions --pred_file_name pred_test.csv \
               --output_model_dir results/debertav3-large/ --interpret_method interaction --ddp --deepspeed --mission output