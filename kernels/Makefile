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

install:
	uv pip install -v --editable . 2>&1 | tee /tmp/ilx-install.log

# install-build-deps:
# 	uv pip install -r ./../requirements/build.txt

wheel:
	uv run -v setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

vllm-wheel:
	cd .. && VLLM_TARGET_DEVICE=empty python setup.py bdist_wheel --dist-dir=kernels/dist

vllm-install:
	cd .. && VLLM_TARGET_DEVICE=empty pip install -v --editable .

serve:
	uv vllm serve "meta-llama/Meta-Llama-3-8B-Instruct" --disable-log-requests -tp 1

test:
	python -c "import vllm._C as c; import vllm.vllm_flash_attn as f; print(c.__file__, f.is_fa_version_supported(3))"
