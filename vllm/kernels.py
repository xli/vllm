# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import pkgutil
import importlib
from vllm.logger import init_logger

import vllm_kernels

logger = init_logger(__name__)


def _alias_modules(package_to_scan, new_parent_name):
    """
    Finds all modules in a given package and aliases them under a new parent.

    Args:
        package_to_scan: The package object to scan for modules (e.g., the 'b' package).
        new_parent_name: The string name of the new parent package (e.g., 'a').
    """

    package_path = package_to_scan.__path__
    package_name_prefix = package_to_scan.__name__ + '.'

    for _, module_name, _ in pkgutil.iter_modules(package_path, prefix=package_name_prefix):
        # Dynamically import the found module
        original_module = importlib.import_module(module_name)

        # Alias the module under the new parent package
        short_name = module_name.split('.')[-1]
        new_module_name = f"{new_parent_name}.{short_name}"

        logger.info("alias module: %s -> %s", module_name, new_module_name)
        sys.modules[new_module_name] = original_module
        setattr(sys.modules[new_parent_name], short_name, original_module)

# Run the aliasing function for the base package 'b'
_alias_modules(vllm_kernels, "vllm")
