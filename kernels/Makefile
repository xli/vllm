MAKEFLAGS += --always-make

# make venv
# source ~/uv_env/vllm-kernels/bin/activate && source ./setup_cuda_env.sh
# make wheel

rm-venv:
	rm -rf ~/uv_env/vllm-kernels

venv:
	uv venv ~/uv_env/vllm-kernels --python "$(shell which python3.12)" --seed

rebuild-venv: rm-venv venv

dev-deps:
	uv pip install -r ./../requirements/dev.txt

install:
	uv pip install -v --editable .

install-build-deps:
	uv pip install -r ./../requirements/build.txt

wheel:
	uv python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

vllm-wheel:
	cd .. && VLLM_TARGET_DEVICE=empty uv python setup.py bdist_wheel --dist-dir=kernels/dist

serve:
	uv vllm serve "meta-llama/Meta-Llama-3-8B-Instruct" --disable-log-requests -tp 1

test:
	uv python -c "import vllm._C as c; import vllm.vllm_flash_attn as f; print(c.__file__, f.is_fa_version_supported(3))"
