# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
import logging
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel

from megatron.core.utils import trace_async_exceptions

from ..__init__ import Request, TypeLookupable
from ..inference import (
    InferenceInterface,
    LLMChatMessage,
    ReturnsRaw,
)

logger = logging.getLogger(__name__)


class AgentBaseModel(BaseModel, extra='allow'):
    pass


class RolloutRequest(Request):
    """Request to agent to generate Rollouts."""

    num_rollouts: int
    inference_interface: InferenceInterface
    validation: bool = False


class GroupedRolloutRequest(Request):
    """Request to agent to generate grouped Rollouts."""

    num_groups: int
    rollouts_per_group: int
    inference_interface: InferenceInterface
    validation: bool = False
    filter_groups_with_same_reward: bool = False


class Rollout(AgentBaseModel):
    """Data for language-based Rollout."""

    trajectory: list[str]
    prompt_length: list[int] | None = None
    reward: float = None
    env_id: str = ''
    problem_id: str | None = None
    policy_staleness: list[list[int]]
    kv_cache_staleness: list[list[int]]
    completed_at_step: list[int]
    num_evictions: list[int]


class TokenRollout(AgentBaseModel):
    """Tokenized representation of a language-based Rollout."""

    trajectory: list[list[int]]
    reward: list[float] | float
    generation_mask: list[list[bool]] | None = None
    logprobs: list[list[float]] | None = None
    env_id: str = ''
    problem_id: str | None = None
    policy_staleness: list[list[int]]
    kv_cache_staleness: list[list[int]]
    completed_at_step: list[int]
    num_evictions: list[int]


class ContrastiveRollout(AgentBaseModel):
    """Contrastive/Preference data for language-based Rollout."""

    chosen_trajectory: list[str]
    rejected_trajectory: list[str]


class Head2HeadRolloutRequest(Request):
    num_rollouts: int
    inference_interface: list[InferenceInterface]
    validation: bool = False


class EvaluationRequest(Request):
    """Request to evaluate N prompts, optionally distributed across ranks."""

    inference_interface: InferenceInterface
    num_prompts: int
    rank_info: tuple[int, int] | None = (
        None  # (rank, total_ranks) if distributed, None for full evaluation
    )
    validation: bool = True


class EvaluationResult(AgentBaseModel):
    prompt: str | list[LLMChatMessage]
    response: str | LLMChatMessage


class RewardEvaluationResult(EvaluationResult):
    reward: float
    problem_id: str | None = None


T = TypeVar('T', bound=EvaluationResult)


class EvaluationResponse(AgentBaseModel, TypeLookupable, Generic[T]):
    env_id: str
    results: list[T]

    def metrics(self):
        raise NotImplementedError(f"{type(self)} did not provide metric aggregation.")


class Agent(ABC, AgentBaseModel):
    pass


class RolloutGenerator(Agent, ABC):
    """An agent that produces Rollout objects containing rollout string and associated reward."""

    @abstractmethod
    async def rollout(self, request: RolloutRequest) -> Rollout: ...

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        return await asyncio.gather(
            *[self.rollout(request=request) for _ in range(request.num_rollouts)]
        )


class ContrastiveRolloutGenerator(Agent, ABC):
    """An agent that produces ContrastiveRollout objects containing two rollout strings, one chosen and one rejected."""

    @abstractmethod
    async def get_contrastive_rollouts(
        self, request: RolloutRequest
    ) -> list[ContrastiveRollout]: ...


class TokenizedRolloutGenerator(Agent, ABC):
    """An agent that produces TokenRollout objects containing rollout token ids and associated rewards.

    Optionally can also provide generation masks to indicate which tokens were generated and token masks to indicate which
    tokens were possible at any given step.
    """

    @abstractmethod
    async def rollout(self, request: RolloutRequest) -> TokenRollout: ...

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        return await asyncio.gather(
            *[self.rollout(request=request) for _ in range(request.num_rollouts)]
        )


class GroupedRolloutGenerator(Agent, ABC):
    """An interface to return grouped Rollout objects to support algorithms like GRPO."""

    parallel_generation_tasks: int = 512
    buffer_size: int = 10

    def __init__(
        self,
        *,
        parallel_generation_tasks: int | None = None,
        buffer_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if parallel_generation_tasks is not None:
            self.parallel_generation_tasks = parallel_generation_tasks
        if buffer_size is not None:
            self.buffer_size = buffer_size
        elif parallel_generation_tasks is not None:
            # In partial-rollout mode, a tiny queue starves producers and collapses
            # decode concurrency into long-tail 1-3 active requests. Keep at least
            # one queue slot per configured generation task.
            self.buffer_size = max(self.buffer_size, parallel_generation_tasks)

    @abstractmethod
    async def group_rollout(self, request: GroupedRolloutRequest) -> list[Rollout]: ...

    async def get_grouped_rollouts(self, request: GroupedRolloutRequest):
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        # If num_groups is -1, we generate a stream of groups.
        # The buffer size is used to create backpressure for each agent in order to balance group generation in a multi-task setting.
        grouped_rollouts: asyncio.Queue[list[Rollout]] = asyncio.Queue(
            maxsize=self.buffer_size if request.num_groups < 0 else 0
        )
        submitted_groups = 0
        yielded_groups = 0
        filtered_groups_dropped = 0

        logger.info(
            "[mrl-grouped-rollouts] start num_groups=%s rollouts_per_group=%s parallel_generation_tasks=%s buffer_size=%s streaming=%s",
            request.num_groups,
            request.rollouts_per_group,
            self.parallel_generation_tasks,
            self.buffer_size if request.num_groups < 0 else 0,
            request.num_groups < 0,
        )

        @trace_async_exceptions(verbose=True)
        async def group_task(task_idx: int):
            nonlocal submitted_groups, filtered_groups_dropped
            logger.info(
                "[mrl-grouped-rollouts] producer_start task=%s submitted_groups=%s qsize=%s",
                task_idx,
                submitted_groups,
                grouped_rollouts.qsize(),
            )
            while request.num_groups == -1 or submitted_groups < request.num_groups:
                submitted_groups += 1
                current_group_idx = submitted_groups
                logger.info(
                    "[mrl-grouped-rollouts] producer_request task=%s group_index=%s qsize_before=%s",
                    task_idx,
                    current_group_idx,
                    grouped_rollouts.qsize(),
                )
                group = await self.group_rollout(request=request)
                logger.info(
                    "[mrl-grouped-rollouts] producer_response task=%s group_index=%s rollout_count=%s qsize_before_put=%s",
                    task_idx,
                    current_group_idx,
                    len(group),
                    grouped_rollouts.qsize(),
                )
                if (
                    not request.filter_groups_with_same_reward
                    or np.std([r.reward for r in group]) > 1e-6
                ):
                    await grouped_rollouts.put(group)
                    logger.info(
                        "[mrl-grouped-rollouts] producer_put task=%s group_index=%s qsize_after_put=%s",
                        task_idx,
                        current_group_idx,
                        grouped_rollouts.qsize(),
                    )
                else:
                    submitted_groups -= 1
                    filtered_groups_dropped += 1
                    logger.info(
                        "[mrl-grouped-rollouts] producer_filtered task=%s group_index=%s filtered_groups_dropped=%s qsize=%s",
                        task_idx,
                        current_group_idx,
                        filtered_groups_dropped,
                        grouped_rollouts.qsize(),
                    )
            logger.info(
                "[mrl-grouped-rollouts] producer_end task=%s submitted_groups=%s filtered_groups_dropped=%s qsize=%s",
                task_idx,
                submitted_groups,
                filtered_groups_dropped,
                grouped_rollouts.qsize(),
            )

        tasks = [asyncio.create_task(group_task(task_idx)) for task_idx in range(self.parallel_generation_tasks)]

        try:
            while grouped_rollouts.qsize() > 0 or not all(task.done() for task in tasks):
                live_group_tasks = sum(1 for task in tasks if not task.done())
                logger.info(
                    "[mrl-grouped-rollouts] consumer_wait qsize=%s submitted_groups=%s yielded_groups=%s live_group_tasks=%s filtered_groups_dropped=%s",
                    grouped_rollouts.qsize(),
                    submitted_groups,
                    yielded_groups,
                    live_group_tasks,
                    filtered_groups_dropped,
                )
                group = await grouped_rollouts.get()
                yielded_groups += 1
                live_group_tasks = sum(1 for task in tasks if not task.done())
                logger.info(
                    "[mrl-grouped-rollouts] consumer_yield qsize_after_get=%s submitted_groups=%s yielded_groups=%s live_group_tasks=%s filtered_groups_dropped=%s rollout_count=%s",
                    grouped_rollouts.qsize(),
                    submitted_groups,
                    yielded_groups,
                    live_group_tasks,
                    filtered_groups_dropped,
                    len(group),
                )
                yield group
        finally:
            for task in tasks:
                task.cancel()
            logger.info(
                "[mrl-grouped-rollouts] end submitted_groups=%s yielded_groups=%s filtered_groups_dropped=%s final_qsize=%s",
                submitted_groups,
                yielded_groups,
                filtered_groups_dropped,
                grouped_rollouts.qsize(),
            )


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
