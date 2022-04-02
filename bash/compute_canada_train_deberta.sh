# !/bin/bash
# SBATCH --gres=gpu:v100l:2      # Request GPU "generic resources"
# SBATCH --cpus-per-task=12  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
# SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
# SBATCH --time=2-0:00
# SBATCH --output=train_debertava-large.out

module load python
export HF_HOME="/project/def-carenini/liraymo6"
source /home/liraymo6/virtualenvs/commonsense/bin/activate

export DATA_DIR="data/"
export OUTPUT_DIR="results/debertav3-large/"
export TOKENIZERS_PARALLELISM=false
export LOADMODEL_ERROR=0
export CUDA_VISIBLE_DEVICES=0,1

deepspeed task.py --data_version csqa_ret_3datasets --append_descr 1 --append_retrieval 1 --append_triples --append_answer_text 1 \
               --preset_model_type debertav3-large --optimizer_type adamw --lr 4e-6 --weight_decay 0.01 --warmup_proportion 0.1 --max_seq_length 512 \
               --batch_size 2 --gradient_acc_step 24 --num_train_epochs 10 --save_interval_step 200 --print_loss_step 10 --print_number_per_epoch 2  \
               --continue_train --clear_output_folder --vary_segment_id  --seed 42 --local_rank 0 --fp16 0 --save_every \
               --output_model_dir results/debertav3-large/ --ddp --deepspeed