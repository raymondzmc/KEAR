export DATA_DIR="data/"
export OUTPUT_DIR="results/test"
export TOKENIZERS_PARALLELISM=false
export LOADMODEL_ERROR=0
export CUDA_VISIBLE_DEVICES=0


python task.py --data_version csqa_ret_3datasets --append_descr 1 --append_retrieval 1 --append_triples --append_answer_text 1 \
               --mission output  --predict_dir results/test/prediction --pred_file_name pred_test.csv --predict_dev \
               --preset_model_type debertav3-base --optimizer_type adamw --lr 4e-6 --weight_decay 0.01 --warmup_proportion 0.1 --max_seq_length 50 \
               --batch_size 1 --gradient_acc_step 48 --num_train_epochs 10 --save_interval_step 100 --print_loss_step 10 --print_number_per_epoch 2  \
               --continue_train --clear_output_folder --vary_segment_id  --seed 42 --local_rank 0 \
               --output_model_dir results/test/ --break_input --interpret_method interaction