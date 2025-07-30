# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""

from typing import Union

import torch
from torch import nn
from torchao.swizzle.swizzle_tensor import SwizzleTensor

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def swizzle_op_weights(
        x: Union[torch.nn.Parameter, torch.nn.Module]) -> torch.nn.Parameter:
    logger.info("swizzle_op_weights: %s", type(x))
    if isinstance(x, torch.nn.Parameter):
        logger.info("Parameter data shape: %s", x.data.shape)
        x = torch.nn.Parameter(SwizzleTensor(x.data.contiguous()))
    elif isinstance(x, (RowParallelLinear, ColumnParallelLinear)):
        assert isinstance(x.weight, torch.nn.Parameter)
        weight = x.weight
        del x.weight
        new_weight = swizzle_op_weights(weight)
        # see vllm/model_executor/layers/linear.py
        # UnquantizedLinearMethod#create_weights
        set_weight_attrs(new_weight, {"input_dim": 1, "output_dim": 0})
        x.register_parameter("weight", new_weight)
    else:
        raise RuntimeError(f"Unrecognized weight type {type(x)}")
    return x


def swizzlfy_llama_mlp(m: nn.Module) -> None:
    # both are attributes of LlamaMLP
    # w13 => language_model.model.layers.\1.feed_forward.gate_up_proj.weight
    m.gate_up_proj = swizzle_op_weights(m.gate_up_proj)
    # w2 => language_model.model.layers.\1.feed_forward.down_proj.weight
    m.down_proj = swizzle_op_weights(m.down_proj)


def swizzlfy(model: nn.Module) -> None:
    logger.info("swizzlfy starts")
    m = model.get_language_model().model
    for layer in m.layers:
        ff = layer.feed_forward
        ffname = str(type(ff))
        logger.info("layer: %s feed_forward: %s", type(layer), ffname)
        if "Llama4MoE" in ffname:
            assert hasattr(ff, "shared_expert")
            assert "LlamaMLP" in str(ff.shared_expert.__class__)
            swizzlfy_llama_mlp(layer.feed_forward.shared_expert)
        elif "LlamaMLP" in ffname:
            swizzlfy_llama_mlp(layer.feed_forward)
        else:
            raise RuntimeError(
                f"unknown layer {type(layer)} feed_forward: {ffname}")

        # Attention linear
        # language_model.model.layers.\1.self_attn.qkv_proj.weight > wqkv
        layer.self_attn.qkv_proj = swizzle_op_weights(layer.self_attn.qkv_proj)
        #language_model.model.layers.\1.self_attn.o_proj.weight > wo
        layer.self_attn.o_proj = swizzle_op_weights(layer.self_attn.o_proj)
