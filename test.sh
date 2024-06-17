python3  main.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks  humaneval \
    --max_length_generation 512 \
    --max_new_tokens 64 \
    --temperature 0.8   \
    --do_sample True  \
    --n_samples 15  \
    --batch_size 4 \
    --trust_remote_code \
    --limit 20 \
    --precision fp16 \
    --allow_code_execution \
    --num_shots 1 \
    --draft meta-llama/Llama-2-7b-hf \
    --max_memory_per_gpu auto \
    --save_generations \
    --save_generations_path humaneval_70B.json

# --act_sparse_type griffin \
#     --act_sparsity 0.5 \
# pal-gsm8k-greedy

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
#     --model meta-llama/Llama-2-70b-chat-hf \
#     --tasks humaneval\
#     --max_length_generation 200 \
#     --temperature 0.8   \
#     --do_sample True  \
#     --n_samples 1  \
#     --batch_size 1  \
#     --trust_remote_code \
#     --allow_code_execution \
#     --limit 5\
#     --precision fp16\
#     --save_generations \
#     --save_generations_path human_eval_70B.json