# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import socket
from typing import AsyncGenerator

import httpx
import yaml
from fastapi import FastAPI
from pydantic import Field, PrivateAttr
from typing_extensions import Self
from uvicorn import Config, Server
from uvicorn.config import LOGGING_CONFIG

LOGGING_CONFIG['root'] = {"handlers": ["default"], "level": "INFO"}

from ... import import_class, inference
from ...agent.api import (
    Agent,
    ContrastiveRollout,
    ContrastiveRolloutGenerator,
    EvaluationAgent,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from ...server.api import (
    EnvironmentServer,
    InferenceServer,
    RemoteEvaluationRequest,
    RemoteGroupedRolloutRequest,
    RemoteRolloutRequest,
)
from .. import agent
from ..api import EnvironmentServer, InferenceServer, RemoteEvaluationRequest, RemoteRolloutRequest


def _safe_unwrap(interface):
    """Unwrap inference interface, falling back to the raw object if type is unregistered."""
    try:
        return interface.unwrap()
    except (KeyError, AttributeError):
        return interface


@EnvironmentServer.register_subclass
class FastAPIEnvServer(EnvironmentServer):
    server_type: str = Field('FastAPIEnvServer', frozen=True, Literal=True)
    env_server_host_port: str
    _server_task: asyncio.Task = PrivateAttr(None)

    @classmethod
    async def launch(cls, env_cls: type[Agent], cls_args: dict, port: int, **kwargs) -> Self:

        app = FastAPI()

        if issubclass(env_cls, GroupedRolloutGenerator):

            @app.post("/grouped_rollouts/")
            async def grouped_rollouts(
                request: RemoteGroupedRolloutRequest,
            ) -> list[list[TokenRollout]]:
                print(f"[ENV /grouped_rollouts/] groups={request.num_groups} host={getattr(request.inference_interface, 'host', '?')}:{getattr(request.inference_interface, 'port', '?')}", flush=True)
                env = env_cls(**cls_args)
                request.inference_interface = _safe_unwrap(request.inference_interface)
                return await env.get_grouped_rollouts(request)

        if issubclass(env_cls, ContrastiveRolloutGenerator):

            @app.post("/contrastive_rollouts/")
            async def contrastive_rollouts(
                request: RemoteRolloutRequest,
            ) -> list[ContrastiveRollout]:
                env = env_cls(**cls_args)
                request.inference_interface = _safe_unwrap(request.inference_interface)
                return await env.get_contrastive_rollouts(request)

        if issubclass(env_cls, RolloutGenerator):

            @app.post("/rollouts/")
            async def rollouts(request: RemoteRolloutRequest) -> list[TokenRollout]:
                env = env_cls(**cls_args)
                request.inference_interface = _safe_unwrap(request.inference_interface)
                return await env.get_reward_rollouts(request)

        if issubclass(env_cls, EvaluationAgent):

            @app.post("/evaluation/")
            async def run_evaluation(request: RemoteEvaluationRequest):
                env = env_cls(**cls_args)
                request.inference_interface = _safe_unwrap(request.inference_interface)
                return await env.run_evaluation(request)

        loop = asyncio.get_event_loop()
        config = Config(app=app, loop=loop, host='0.0.0.0', port=port)
        server = Server(config)
        server_task = loop.create_task(server.serve())

        ip = socket.gethostbyname(socket.gethostname())

        launched_server = cls(env_server_host_port=f"{ip}:{config.port}", **kwargs)
        launched_server._server_task = server_task

        return launched_server

    def kill(self):
        return self._server_task.cancel()

    async def get_contrastive_rollouts(self, request: RolloutRequest) -> list[ContrastiveRollout]:
        assert isinstance(
            request.inference_interface, InferenceServer
        ), "Rollout requests to remote server must contain an InferenceServer object"
        payload = request.model_dump()
        payload["inference_interface"] = request.inference_interface.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/contrastive_rollouts/",
                json=payload,
                timeout=None,
            )
        rollouts = [ContrastiveRollout.model_validate(r) for r in response.json()]
        return rollouts

    async def group_rollout(self, request: GroupedRolloutRequest):
        assert (
            False
        ), "Calling group_rollout on FastAPIEnvServer is not supported, use get_grouped_rollouts"

    async def get_grouped_rollouts(
        self, request: GroupedRolloutRequest
    ) -> AsyncGenerator[list[TokenRollout], None]:
        print(f"[GPU->ENV] get_grouped_rollouts called: num_groups={request.num_groups} "
              f"rollouts_per_group={request.rollouts_per_group} "
              f"iface_type={type(request.inference_interface).__name__} "
              f"target={self.env_server_host_port}", flush=True)
        assert isinstance(
            request.inference_interface, InferenceServer
        ), f"Rollout requests to remote server must contain an InferenceServer object, got {type(request.inference_interface)}"
        assert request.num_groups != -1, "FastAPIEnvServer does not support group rollout streaming"
        payload = request.model_dump()
        iface_dump = request.inference_interface.model_dump()
        payload["inference_interface"] = iface_dump
        print(f"[GPU->ENV] POST http://{self.env_server_host_port}/grouped_rollouts/ groups={request.num_groups} payload_keys={list(payload.keys())}", flush=True)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/grouped_rollouts/", json=payload, timeout=None
            )
        print(f"[GPU->ENV] Response: {response.status_code} len={len(response.content)}", flush=True)
        if response.status_code != 200:
            print(f"[GPU->ENV] ERROR: Server returned {response.status_code}: "
                  f"{response.text[:500]}", flush=True)
            raise RuntimeError(
                f"Environment server returned HTTP {response.status_code} for grouped_rollouts"
            )
        rollouts = [[TokenRollout.model_validate(r) for r in group] for group in response.json()]
        print(f"[GPU->ENV] Parsed {len(rollouts)} rollout groups", flush=True)
        for rollout in rollouts:
            yield rollout

    async def rollout(self, request: RolloutRequest) -> TokenRollout:
        assert (
            False
        ), "Calling rollout on FastAPIEnvServer is not supported, use get_reward_rollouts"

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        assert isinstance(
            request.inference_interface, InferenceServer
        ), "Rollout requests to remote server must contain an InferenceServer object"
        payload = request.model_dump()
        payload["inference_interface"] = request.inference_interface.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/rollouts/", json=payload, timeout=None
            )
        rollouts = [TokenRollout.model_validate(r) for r in response.json()]
        return rollouts

    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse:
        assert isinstance(
            request.inference_interface, InferenceServer
        ), "Evaluation requests to remote server must contain an InferenceServer object"
        payload = request.model_dump()
        payload["inference_interface"] = request.inference_interface.model_dump()
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/evaluation/", json=payload, timeout=None
            )
        response = EvaluationResponse.model_validate(response.json()).unwrap()
        return response


def run(agent_cls: type[Agent], cls_args: dict, port: int):
    loop = asyncio.new_event_loop()

    async def run_server():
        server: FastAPIEnvServer = await FastAPIEnvServer.launch(
            env_cls=agent_cls, cls_args=cls_args, port=port
        )
        print(server.model_dump(exclude={'_server_task'}))
        await server._server_task

    loop.run_until_complete(run_server())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with open(args.env_config, 'r') as f:
        config = yaml.safe_load(f)[0]
    agent_cls = import_class(config['agent_type'])
    cls_args = config['agent_args']
    run(agent_cls, cls_args, port=args.port)
