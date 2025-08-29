# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator
from typing import Optional, Union

from fastapi import Request

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              IOProcessorRequest,
                                              IOProcessorResponse)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.utils import merge_async_iterators

logger = init_logger(__name__)


class ServingPoolingWithIOPlugin(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        vllm_config: VllmConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=vllm_config.model_config,
            models=models,
            request_logger=request_logger,
        )
        io_processor_plugin = self.model_config.io_processor_plugin
        self.io_processor = get_io_processor(vllm_config, io_processor_plugin)

    async def create_pooling_with_io_plugin(
        self,
        request: IOProcessorRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[IOProcessorResponse, ErrorResponse]:

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"io-processor-{self._base_request_id(raw_request)}"

        try:

            pooling_params = request.to_pooling_params()
            trace_headers = (None if raw_request is None else await
                             self._get_trace_headers(raw_request.headers))

            if self.io_processor is None:
                raise ValueError(
                    "No IOProcessor plugin installed. Please refer "
                    "to the documentation and to the "
                    "'prithvi_geospatial_mae_io_processor' "
                    "offline inference example for more details.")

            validated_prompt = self.io_processor.parse_request(request)

            # Here I am assuming that the image prediction request might
            # be split in multiple prompts because of tiling
            prompts = await self.io_processor.pre_process_async(
                prompt=validated_prompt, request_id=request_id)

            # Schedule the request and get the result generator.
            # Note that at the moment, models capable of generating images
            # are piggybacking on the pooling models support.
            # See the PrithviMAEGeospatial model
            generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

            for i, prompt in enumerate(prompts):
                request_id_item = f"{request_id}-{i}"

                generator = self.engine_client.encode(
                    prompt,
                    pooling_params,
                    request_id_item,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )
                generators.append(generator)

            output = await self.io_processor.post_process_async(
                model_output=merge_async_iterators(*generators),
                request_id=request_id,
            )

            return self.io_processor.output_to_response(output)

        except ValueError as e:
            return self.create_error_response(str(e))
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
