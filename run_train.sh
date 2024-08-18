python train.py \
--model_name meta-llama/Meta-Llama-3.1-8B \
--precision fp16_autocast \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 2048 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing True \
--use_cpu_offload True \
# --log_to wandb \
--dataset alpaca \
--verbose False \
--save_model True \
--verbose True  \
--output_dir /scratch/tathagato/fsdp_qlora_experiments \
| tee output.txt