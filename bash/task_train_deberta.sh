export DATA_DIR="data/"
export OUTPUT_DIR="test/"
export TOKENIZERS_PARALLELISM=false
export LOADMODEL_ERROR=0

# deepspeed task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --lr 5e-6 \
# 		--append_answer_text 1 --weight_decay 0 --preset_model_type debertav2-xlarge --batch_size 1 --max_seq_length 50 
# 		--num_train_epochs 15 --save_interval_step 4 --continue_train --print_number_per_epoch 1 --vary_segment_id --seed 42 \
# 		--warmup_proportion 0.1 --optimizer_type adamw --ddp --deepspeed --test_mode --clear_output_folder


# python -m torch.distributed.launch --nproc_per_node=2 task.py --append_descr 1 \
# --data_version csqa_ret_3datasets --lr 1e-5 --append_answer_text 1 --weight_decay 0.01 \
#  --preset_model_type debertav3 --batch_size 2 --max_seq_length 50 --num_train_epochs 10 --save_interval_step 2 \
#  --continue_train --print_number_per_epoch 2 --vary_segment_id --seed 42 --warmup_proportion 0.1 --optimizer_type adamw --ddp --print_loss_step 10 --clear_output_folder

# DeBERTa LR: {4e−6, 6e−6, 9e−6}
# VAT:  α ∈ {0.1, 1.0, 10.0}, ε = 1e−5

python task.py --data_version csqa_ret_3datasets --append_descr 1 --append_triples --append_retrieval 1  --append_answer_text 1 \
			   --preset_model_type debertav3 --optimizer_type adamw --lr 4e-6 --weight_decay 0.01 --warmup_proportion 0.1 --max_seq_length 50 \
			   --batch_size 48 --num_train_epochs 10 --save_interval_step 2 --print_loss_step 10 --print_number_per_epoch 2  \
			   --continue_train --clear_output_folder --vary_segment_id  --seed 42