MAKEFLAGS += --always-make

# make venv
# source .venv/bin/activate && source ./setup_cuda_env.sh
# make wheel

rm-venv:
	rm -rf .venv

venv:
	uv venv --python "$(shell which python3.12)" --seed --link-mode=copy

rebuild-venv: rm-venv venv

dev-deps:
	uv pip install -r ./../requirements/dev.txt

install-build-deps:
	uv pip install -r ./../requirements/build.txt

install:
	uv pip install -v --no-build-isolation --editable . 2>&1 | tee /tmp/ilx-install.log

wheel:
	pip wheel -v --no-build-isolation --wheel-dir=dist . 2>&1 | tee /tmp/ilx-wheel.log

vllm-wheel:
	cd .. && VLLM_TARGET_DEVICE=empty pip wheel -v --wheel-dir=kernels/dist .

vllm-install:
	cd .. && VLLM_TARGET_DEVICE=empty pip install -v --editable .

serve:
	vllm serve "meta-llama/Meta-Llama-3-8B-Instruct" --disable-log-requests -tp 1

test:
	python -c "import vllm._C as c; import vllm.vllm_flash_attn as f; print(c.__file__, f.is_fa_version_supported(3))"
