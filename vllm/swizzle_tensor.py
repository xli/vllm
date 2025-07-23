# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""

import torch
from torch import nn
from torchao.swizzle.swizzle_tensor import SwizzleTensor

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)

logger = init_logger(__name__)


def swizzle_op_weights(x: torch.nn.Parameter, ) -> torch.nn.Parameter:
    if isinstance(x, torch.nn.Parameter):
        logger.info(
            "Swizzled Parameter %s for bf16 GEMM, type: %s, device: %s",
            x.data.shape, type(x.data), x.data.device)
        x = torch.nn.Parameter(SwizzleTensor(x.data.contiguous()))
        logger.info("After swizzled type: %s, device: %s", type(x.data),
                    x.data.device)
    elif isinstance(x, (RowParallelLinear, ColumnParallelLinear)):
        logger.info(
            "Swizzled parallel linear %s for bf16 GEMM, type: %s, device: %s",
            x.weight.data.shape, type(x.weight.data), x.weight.data.device)
        x.weight = torch.nn.Parameter(SwizzleTensor(
            x.weight.data.contiguous()))
        logger.info("After swizzled type: %s, device: %s", type(x.weight.data),
                    x.weight.data.device)
    else:
        logger.info("Unrecognized weight type %s for bf16 Swizzling GEMM",
                    type(x))
    return x


def swizzlfy(model: nn.Module) -> None:
    m = model.get_language_model().model
    logger.info("process_weights_after_loading starts")
    logger.info(
        "process_weights_after_loading: start layer: %s, end layer: %s",
        m.start_layer.__class__, m.end_layer.__class__)
    for block in m.layers:
        logger.info("Layer %s, feed_forward class: %s", block.__class__,
                    block.feed_forward.__class__)
        if "Llama4MoE" in str(block.feed_forward.__class__):
            logger.info("Llama4MoE")
            # logger.info("Llama4MoE experts w13_weight")
            # block.feed_forward.experts.w13_weight = swizzle_op_weights(
            #         block.feed_forward.experts.w13_weight)

            # logger.info("Llama4MoE experts w2_wight (w_out_shared_DF)")
            # block.feed_forward.experts.w2_weight = swizzle_op_weights(
            #         block.feed_forward.experts.w2_weight)
            logger.info("Llama4MoE shared_expert gate_up_proj")
            block.feed_forward.shared_expert.gate_up_proj = swizzle_op_weights(
                block.feed_forward.shared_expert.gate_up_proj)

            logger.info("Llama4MoE shared_expert down_proj")
            block.feed_forward.shared_expert.down_proj = swizzle_op_weights(
                block.feed_forward.shared_expert.down_proj)
            logger.info("Llama4MoE end")
        elif "LlamaMLP" in str(block.feed_forward.__class__):
            logger.info("LlamaMLP")

            logger.info("LlamaMLP gate_up_proj (w13)")
            block.feed_forward.gate_up_proj = swizzle_op_weights(
                block.feed_forward.gate_up_proj)

            logger.info("LlamaMLP down_proj (w2)")
            # language_model.model.layers.\1.feed_forward.down_proj.weight
            block.feed_forward.down_proj = swizzle_op_weights(
                block.feed_forward.down_proj)
        else:
            logger.info("unknown: %s", block.feed_forward.__class__)

        logger.info("self attn qkv_proj")
        # Attention linear
        # language_model.model.layers.\1.self_attn.qkv_proj.weight > wqkv
        block.self_attn.qkv_proj = swizzle_op_weights(block.self_attn.qkv_proj)
        logger.info("self attn o_proj")
        #language_model.model.layers.\1.self_attn.o_proj.weight > wo
        block.self_attn.o_proj = swizzle_op_weights(block.self_attn.o_proj)
