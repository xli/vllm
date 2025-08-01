MAKEFLAGS += --always-make

# source /home/ilx/uv_env/vllm/bin/activate

request:
	curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
		-d '{"messages": [{"role": "user", "content": "Hello, vLLM!"}],"max_tokens": 100}' | jq .

serve-l4: v=1
serve-l4:
	VLLM_TORCH_PROFILER_DIR=~/vllm_profile \
	VLLM_USE_TRITON_FLASH_ATTN=1 \
	FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
	VLLM_ROCM_USE_AITER=1 \
	VLLM_FP8_PADDING=1 \
	SAFETENSORS_FAST_GPU=1 \
	VLLM_USE_V1=$(v) \
	vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 -tp 8 --disable-log-requests | tee /tmp/vllm-l4.log

serve-ds: v=1
serve-ds:
	VLLM_TORCH_PROFILER_DIR=~/vllm_profile \
	VLLM_USE_TRITON_FLASH_ATTN=1 \
	FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
	VLLM_ROCM_USE_AITER=0 \
	VLLM_FP8_PADDING=1 \
	SAFETENSORS_FAST_GPU=1 \
	VLLM_USE_V1=$(v) \
	VLLM_GPU_MEMORY_UTILIZATION=0.95 \
	vllm serve deepseek-ai/DeepSeek-V3 -tp 8 --disable-log-requests | tee /tmp/vllm-ds.log

serve_aiter: v=1
serve_aiter:
	VLLM_USE_TRITON_FLASH_ATTN=1 \
	FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
	VLLM_ROCM_USE_AITER=1 \
	VLLM_FP8_PADDING=1 \
	SAFETENSORS_FAST_GPU=1 \
	VLLM_USE_V1=$(v) \
	VLLM_GPU_MEMORY_UTILIZATION=0.95 \
	vllm serve deepseek-ai/DeepSeek-V3 -tp 8 --max-num-seqs 256 --block-size 1 | tee /tmp/vllm-ds.log
# --block-size 1

eval: tasks=gsm8k
eval: bs=16
eval: cc=2
eval:
	FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE lm_eval --model local-completions --tasks $(tasks) \
		--model_args model=deepseek-ai/DeepSeek-V3,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=$(cc),max_retries=3,tokenized_requests=False \
		--batch_size $(bs) \
		| tee /tmp/vllm-ds-eval.log

eval-dsx:
	cd /data/users/ilx/fbsource/fbcode && \
		buck run @mode/opt //gen_ai/evals:thrift_main -- \
		--tokenizer-path=manifold://fair_llms/tree/70b-v2/tokenizer_final_32k.minus_inf_ws.model \
      	--debug \
      	--tasks=gsm8k.8_shot.1_gen \
      	--request-chat-mode=true \
      	--top-p=1.0 \
      	--top-k=0 \
      	--max-samples=200 \
      	--max-concurrent-requests=4 \
		--vllm \
		--server-ip 127.0.0.1 \
		--server-port 8000 | tee /tmp/vllm-dsx-eval.log


benchmark: cc=16
benchmark:
	FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE vllm bench serve \
		--model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
		--host 127.0.0.1 \
		--dataset-name random \
		--ignore-eos \
		--num-prompts 640 \
		--random-input-len 10000 \
		--random-output-len 1000 \
		--max-concurrency $(cc)

pyspy: pid=3805529
pyspy:
	py-spy record -p $(pid)  -o pyspy.svg -d 60 -r 50
