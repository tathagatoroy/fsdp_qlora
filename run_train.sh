python train.py \
--model_name meta-llama/Meta-Llama-3.1-8B \
--precision bf16_buffers_autocast \
--gradient_accumulation_steps 2 \
--batch_size 2 \
--context_length 2048 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing True \
--use_cpu_offload True \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir /scratch/tathagato/fsdp_qlora_experiments