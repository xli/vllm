MAKEFLAGS += --always-make


# source /home/ilx/uv_env/vllm/bin/activate

serve:
	VLLM_USE_TRITON_FLASH_ATTN=1 \
	FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
	VLLM_ROCM_USE_AITER=1 \
	VLLM_FP8_PADDING=1 \
	VLLM_USE_V1=1 \
	SAFETENSORS_FAST_GPU=1 \
	vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size 8 | tee /tmp/vllm-swizzle.log

request:
	curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
		-d '{"messages": [{"role": "user", "content": "Hello, vLLM!"}],"max_tokens": 100}' | jq .
