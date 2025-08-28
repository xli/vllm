# vLLM kernels

Made up of vLLM kernels + custom ops + C extension dependencies like vllm-flash-attn


## Overview

* vllm/kernels.py imports vllm_kernels package and creates alias modules to keep backward compatible, e.g. vllm.\_C => vllm_kernels.\_C
    * Ignores import error before we release a new version of vllm that is depending on vllm_kernels package
    * Won't overwrite module if the module exists.
* vllm_kernels is a namespace package for future extension: allow vendor managed package to install new vllm_kernels module.

## New directory structure

vllm:
    - vllm: vllm engine python code
    - pyproject.toml: vllm package
    - kernels
        - pyproject.toml: vllm-kernels package
        - csrc: C extension source code


## Adoption Plan

* refactoring: remove vllm/vllm_flash_attn for creating alias vllm.vllm_flash_attn => vllm_kernels.vllm_flash_attn
    * existing build process (build vllm with exts or precompile exts) should work as usual.
* refactoring: add build_destination to cmake files, so that we can customize build destination in vllm_kernels/setup.py
* create kernels directory for build new vllm-kernels package: this change only adds new package building process without changing anything existing
	* the directories / files that are duplicated with the ones under vllm will be using symbol links
	* in this change, we can install them separately by:
		* build vllm-kernels wheel
		* build vllm wheel with VLLM_TARGET_DEVICE=empty
	* for kernels development, new workflow works:
		* make install-dev: installs both vllm-kernels and vllm as dev package (by pip install --editable .)
* add CI build and release new vllm-kernels package.
    * vllm-kernels will start with using same version number with vllm package, so that it is easy to identify same version packages.
    * make a release for v0.10.1.1
    * Ideally, vllm-kernels package should be backward compatible with vllm package.
    * Add compatiability test in CI, make sure vllm-kernels is backward compatible with vllm starting from v0.10.1.1
* Move kernels tests into kernels directory
* Once new vllm-kernels package is released, update vllm package to only package python code, equivalent to build the vllm package with VLLM_TARGET_DEVICE=empty
	* Also remove cmake and csrc symbol links in the kernels dir, and move directories into kernels dir
