# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import nn

from vllm.config import DeviceConfig, VllmConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.neuron import get_neuron_model
from vllm.multimodal import BatchedTensorInputs, MultiModalKwargs
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelInputForNeuron(ModelRunnerInputBase):
    """
    Used by the NeuronModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    input_block_ids: Optional[torch.Tensor] = None
    sampling_metadata: SamplingMetadata = None
    multi_modal_kwargs: BatchedTensorInputs = None
    adapter_ids: Optional[str] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        return {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "input_block_ids": self.input_block_ids,
            "sampling_metadata": self.sampling_metadata,
            "multi_modal_kwargs": self.multi_modal_kwargs,
        }

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForNeuron":
        return ModelInputForNeuron(
            input_tokens=tensor_dict["input_tokens"],
            input_positions=tensor_dict["input_positions"],
            input_block_ids=tensor_dict["input_block_ids"],
            sampling_metadata=tensor_dict["sampling_metadata"],
            multi_modal_kwargs=tensor_dict["multi_modal_kwargs"],
        )


class NeuronModelRunner(ModelRunnerBase[ModelInputForNeuron]):
    """A model runner for AWS Neuron hardware"""

    # NEURON has an upper limit on the top_k
    _MAX_NEURON_SAMPLING_TOP_K = 256

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        ModelRunnerBase.__init__(self, vllm_config)

        if (self.model_config is not None
                and self.model_config.get_sliding_window()):
            logger.warning("Sliding window is not supported on Neuron. "
                           "The model will run without sliding window.")
        self.device_config = (self.device_config if self.device_config
                              is not None else DeviceConfig())
        self.lora_config = vllm_config.lora_config
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

        # Once NEURON_ON_DEVICE_SAMPLING_DISABLED is set to a non-zero value,
        # turn off on-device sampling.
        self._on_device_sampling_disabled = int(
            os.getenv("NEURON_ON_DEVICE_SAMPLING_DISABLED", "0"))

        # NEURON needs to update sampling parameters when request IDs change
        # across batches. This variable stores the previous batch's request IDs
        # to determine if an update is needed.
        self._previous_batch_request_ids: List[str] = []

        if not self._on_device_sampling_disabled:
            self._init_neuron_sampling()

    def _init_neuron_sampling(self) -> None:
        if current_platform.use_transformers_neuronx():
            from transformers_neuronx.config import GenerationConfig
        else:
            from transformers import GenerationConfig
        logger.warning(
            "On-device sampling is turned on in Neuron by default, only "
            "top_k, top_p, and temperature are current supported sampling "
            "parameters. To turn off the on-device sampling, please set "
            "the environment variable NEURON_ON_DEVICE_SAMPLING_DISABLED=1.")
        self.model_config.neuron_sampling_params = GenerationConfig(
            max_length=self.scheduler_config.max_model_len,
            do_sample=True,
            per_batch_line=True,
            top_k=[self._MAX_NEURON_SAMPLING_TOP_K] \
                  * self.scheduler_config.max_num_seqs,
            top_p=[1.0] * self.scheduler_config.max_num_seqs,
            temperature=[1.0] * self.scheduler_config.max_num_seqs,
            dynamic=True,
            global_top_k=self._MAX_NEURON_SAMPLING_TOP_K)

    def load_model(self) -> None:
        self.model = get_neuron_model(self.model_config,
                                      parallel_config=self.parallel_config,
                                      scheduler_config=self.scheduler_config)

    def get_model(self) -> nn.Module:
        return self.model

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int],
               BatchedTensorInputs]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []

        seq_lens: List[int] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(seq_len)))

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            assert len(block_table) == 1
            input_block_ids.append(block_table[0])

            mm_kwargs = seq_group_metadata.multi_modal_data
            if mm_kwargs:
                mm_kwargs = self.process_multi_modal_data_neuron(mm_kwargs)
                multi_modal_kwargs_list.append(mm_kwargs)

        max_seq_len = max(seq_lens)
        assert max_seq_len > 0
        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=max_seq_len,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=max_seq_len,
                                               dtype=torch.long,
                                               device=self.device)
        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return (input_tokens, input_positions, input_block_ids, seq_lens,
                multi_modal_kwargs)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []
        context_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])
                context_lens.append(seq_len)

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                assert len(block_table) == 1
                input_block_ids.append(block_table[0])

        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=1,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=1,
                                               dtype=torch.long,
                                               device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)
        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)

        return input_tokens, input_positions, input_block_ids

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForNeuron:
        return ModelInputForNeuron.from_broadcasted_tensor_dict(tensor_dict)

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNeuron:
        multi_modal_kwargs = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, input_block_ids, seq_lens,
             multi_modal_kwargs
             ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             input_block_ids) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None

        if not self._on_device_sampling_disabled:
            for seq_group_metadata in seq_group_metadata_list:
                sampling_params = seq_group_metadata.sampling_params
                top_k, top_p, temperature = (
                    self._convert_to_neuron_sampling_params(sampling_params))
                sampling_params.top_k = top_k
                sampling_params.top_p = top_p
                sampling_params.temperature = temperature

        # we need multi_modal_data for later tokens as well
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        for seq_group_metadata in seq_group_metadata_list:
            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                multi_modal_kwargs_list.append(mm_data)
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since neuron worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))

        if current_platform.use_transformers_neuronx(
        ) and not self._on_device_sampling_disabled:
            # Once the request IDs are changed in current iteration, we will
            # update the on-device sampling parameters.
            current_batch_request_ids = [
                seq_group_meta_data.request_id
                for seq_group_meta_data in seq_group_metadata_list
            ]
            if current_batch_request_ids != self._previous_batch_request_ids:
                self._update_neuron_sampling_params(seq_group_metadata_list)
                self._previous_batch_request_ids = current_batch_request_ids

        return ModelInputForNeuron(input_tokens=input_tokens,
                                   input_positions=input_positions,
                                   input_block_ids=input_block_ids,
                                   sampling_metadata=sampling_metadata,
                                   multi_modal_kwargs=multi_modal_kwargs)

    def _update_neuron_sampling_params(
            self, seq_group_metadata_list: List[SequenceGroupMetadata]):
        # Update Neuron sampling parameters (GenerationConfig in Neuron)
        current_sampling_params = self.model_config.neuron_sampling_params
        assert current_sampling_params is not None, (
            f"Failed to update sampling_params, "
            f"current sampling params is {current_sampling_params}")

        is_update_needed = False

        top_k = current_sampling_params.top_k
        top_p = current_sampling_params.top_p
        temperature = current_sampling_params.temperature

        # The index of a sequence's sampling parameters in neuron is equal to
        # its index in `input_block_ids`.
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params

            seq_group_top_k = sampling_params.top_k
            seq_group_top_p = sampling_params.top_p
            seq_group_temperature = sampling_params.temperature

            for seq_id in seq_ids:
                index = seq_group_metadata.block_tables[seq_id][0]
                if (top_k[index] != seq_group_top_k
                        or top_p[index] != seq_group_top_p
                        or temperature[index] != seq_group_temperature):
                    is_update_needed = True

                top_k[index] = seq_group_top_k
                top_p[index] = seq_group_top_p
                temperature[index] = seq_group_temperature

        # update_generation_config is only available in transformers-neuronx
        if is_update_needed and current_platform.use_transformers_neuronx():
            self.model.model.update_generation_config(current_sampling_params)

    def _convert_to_neuron_sampling_params(
            self, sampling_params: SamplingParams) -> Tuple[int, float, float]:
        # Returns the top_k, top_p and temperature parameters for neuron.
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        temperature = sampling_params.temperature

        if temperature == 0.0:
            # Enable greedy sampling on zero temperature
            return (1, 1.0, 1.0)
        if top_k < 1 or top_k > self._MAX_NEURON_SAMPLING_TOP_K:
            top_k = self._MAX_NEURON_SAMPLING_TOP_K

        return (top_k, top_p, temperature)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForNeuron,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "NeuronModelRunner does not support multi-step execution.")

        # extract top_k, top_p and temperature from model_input for neuron
        # forward call
        sampling_params = (torch.tensor([[
            seq_group.sampling_params.top_k, seq_group.sampling_params.top_p,
            seq_group.sampling_params.temperature
        ] for seq_group in model_input.sampling_metadata.seq_groups]))

        if current_platform.use_neuronx_distributed():
            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                input_block_ids=model_input.input_block_ids,
                sampling_params=sampling_params,
                adapter_ids=model_input.adapter_ids,
                **MultiModalKwargs.as_kwargs(
                    model_input.multi_modal_kwargs or {},
                    device=self.device,
                ),
            )
        elif current_platform.use_transformers_neuronx():
            # [TODO] validate on-device sampling
            # The model signature may need change for on-device sampling
            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                input_block_ids=model_input.input_block_ids,
                **MultiModalKwargs.as_kwargs(
                    model_input.multi_modal_kwargs or {},
                    device=self.device,
                ),
            )

        # Compute the logits only if the on-device sampling is turned off as
        # on-device sampling outputs the token ids.
        if self._on_device_sampling_disabled:
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)
        else:
            logits = hidden_states

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    def process_multi_modal_data_neuron(self, mm_data):
        # this is a no-op for NeuronModelRunner
        return mm_data

    def remove_all_loras(self):
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def add_lora(self, lora_request: LoRARequest):
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")
