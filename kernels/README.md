# vLLM kernels

Made up of vLLM kernels + custom ops + C extension dependencies like vllm-flash-attn


## Overview

* vllm/kernels.py imports vllm_kernels package and creates alias modules to keep backward compatible, e.g. vllm.\_C => vllm_kernels.\_C
    * Ignores import error before we release a new version of vllm that is depending on vllm_kernels package
    * Won't overwrite module if the module exists.
* vllm_kernels is a namespace package (no `__init__.py`) for future extension: allow vendor managed package to install new vllm_kernels module.

## Directory structure

* vllm:
    * kernels
        * pyproject.toml: vllm-kernels package
        * setup.py
        * csrc: C extension source code
        * cmake
        * CMakeLists.txt
        * vllm_kernels: dir required for building extensions

## Adoption Plan

* refactoring: add build_destination to cmake files, so that we can customize build destination in vllm_kernels/setup.py
* create kernels directory for build new vllm-kernels package: this change adds new package building process without changing existing vllm package build
	* the directories / files that are duplicated with the ones under vllm will be using symbol links
	* in this change, we can build and install vllm and vllm-kernels separately in vllm/kernels directory:
		* install vllm-kernels: uv pip install -v --editable .
		* install vllm package without extensions: cd .. && VLLM_TARGET_DEVICE=empty pip install -v --editable .
        * similarly, we can build wheels for them separately and install them.
	* for kernels development, new workflow works:
		*
* add CI build and release new vllm-kernels package.
    * There are couple options for versioning vllm-kernels package, choosing to use same version number with vllm package for simplicity
    * make a release for v0.10.1.1
    * Ideally, vllm-kernels package should be backward compatible with vllm package.
    * Add compatiability test in CI, make sure vllm-kernels is backward compatible with vllm starting from v0.10.1.1
        * this can be complex, may need to test different hardware and feature combinations
        * we can start with locking vllm and vllm-kernels to use same version
* setup nightly build vllm-kernels package, switch VLLM_USE_PRECOMPILED to use vllm-kernels nightly build
    * probably can change to:
        * when install vllm from source (pip install -e .), we pick the vllm-kernels version like VLLM_USE_PRECOMPILED does
        * when install vllm from pypi (pip install vllm), it picks the dependency version packaged in the wheel.
    * to install vllm-kernels and vllm at same version without cache:
        * build wheels separately at same version and install them.
* Move kernels tests into kernels directory
* Once new vllm-kernels package is released, update vllm package to only package python code, equivalent to build the vllm package with VLLM_TARGET_DEVICE=empty
	* Remove cmake and csrc symbol links in the kernels dir, and move directories into kernels dir
    * update vllm/setup.py to package python code only
