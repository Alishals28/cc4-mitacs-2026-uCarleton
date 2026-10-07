"""Short validation rollouts for independent CC4 Blue policies.

This module executes initialized policies; it is not a PPO/DQN trainer. It
tracks multi-tick actions so the environment is not credited with actions that
the simulator ignored while an earlier action was still running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import torch
from torch import Tensor

from CybORG.Agents.CC4LocalObservation import CC4LocalObservationAdapter
from CybORG.Agents.CC4VariableObservation import CC4VariableHostAdapter
from CybORG.Agents.CC4Representations import (
    AgentActionSpec,
    CC4ModelInput,
    CC4ModelInputBuilder,
    CC4TransformerEncoder,
    DeepSetsEncoder,
    IndependentPolicy,
    VariableCC4DeepSetsEncoder,
    VariableCC4InputBuilder,
    VariableCC4ModelInput,
    VariableCC4TransformerEncoder,
    VariableCC4ZeroPaddingEncoder,
    VariableIndependentPolicy,
    ZeroPaddingEncoder,
)


RepresentationName = Literal["zero_padding", "deep_sets", "transformer"]
VariableRepresentationName = Literal["zero_padding", "deep_sets", "transformer"]


@dataclass(frozen=True)
class AgentDecision:
    """One policy decision (as opposed to an environment tick spent waiting)."""

    environment_step: int
    agent_name: str
    action_index: int
    action_label: str
    action_duration: int
    old_log_probability: float
    model_input: CC4ModelInput | VariableCC4ModelInput
    action_mask: Tensor


@dataclass(frozen=True)
class RolloutStep:
    """Joint environment-tick record; the team reward is stored once here."""

    environment_step: int
    active_agents: tuple[str, ...]
    decisions: tuple[str, ...]
    forced_wait_agents: tuple[str, ...]
    action_indices: dict[str, int]
    action_labels: dict[str, str]
    team_reward: float
    per_agent_rewards: dict[str, float]
    terminated: dict[str, bool]
    truncated: dict[str, bool]


@dataclass
class CC4RolloutResult:
    """Artifacts from a bounded integration rollout, not a trained result."""

    steps: list[RolloutStep] = field(default_factory=list)
    decisions_by_agent: dict[str, list[AgentDecision]] = field(default_factory=dict)
    team_return: float = 0.0


def build_independent_policies(
    wrapper,
    representation: RepresentationName,
    *,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, IndependentPolicy], dict[str, CC4ModelInputBuilder]]:
    """Create a distinct encoder, action head, and input builder per Blue agent."""

    factories = {
        "zero_padding": lambda n: ZeroPaddingEncoder(n),
        "deep_sets": lambda _n: DeepSetsEncoder(),
        "transformer": lambda _n: CC4TransformerEncoder(),
    }
    if representation not in factories:
        raise ValueError(f"Unknown CC4 representation: {representation}")

    policies: dict[str, IndependentPolicy] = {}
    builders: dict[str, CC4ModelInputBuilder] = {}
    for agent in wrapper.possible_agents:
        if "blue" not in agent:
            continue
        adapter = CC4LocalObservationAdapter(wrapper, agent)
        action_spec = AgentActionSpec.from_wrapper(wrapper, agent)
        policy = IndependentPolicy(
            factories[representation](len(adapter.subnets)), action_spec
        ).to(device)
        policies[agent] = policy
        builders[agent] = CC4ModelInputBuilder(adapter)

    if len(policies) != 5:
        raise ValueError(f"Expected five independent CC4 Blue policies, got {len(policies)}")
    if len({id(policy) for policy in policies.values()}) != len(policies):
        raise ValueError("Each Blue agent must have a distinct policy instance")
    return policies, builders


def build_variable_independent_policies(
    wrapper,
    representation: VariableRepresentationName,
    *,
    device: torch.device | str = "cpu",
) -> tuple[
    dict[str, VariableIndependentPolicy],
    dict[str, CC4VariableHostAdapter],
    VariableCC4InputBuilder,
]:
    """Create five independent policies for the variable-host contract."""

    policies: dict[str, VariableIndependentPolicy] = {}
    adapters: dict[str, CC4VariableHostAdapter] = {}
    factories = {
        "zero_padding": lambda adapter: VariableCC4ZeroPaddingEncoder(
            len(adapter.subnets) * 16, len(adapter.subnets)
        ),
        "deep_sets": lambda _adapter: VariableCC4DeepSetsEncoder(),
        "transformer": lambda _adapter: VariableCC4TransformerEncoder(),
    }
    if representation not in factories:
        raise ValueError(f"Unknown variable CC4 representation: {representation}")

    for agent in wrapper.possible_agents:
        adapter = CC4VariableHostAdapter(wrapper, agent)
        action_spec = AgentActionSpec.from_wrapper(wrapper, agent)
        policies[agent] = VariableIndependentPolicy(
            factories[representation](adapter), action_spec
        ).to(device)
        adapters[agent] = adapter

    if len(policies) != 5 or len({id(policy) for policy in policies.values()}) != 5:
        raise ValueError("Expected five independent variable-host policies")
    return policies, adapters, VariableCC4InputBuilder()


class CC4RolloutRunner:
    """Run a short joint rollout and validate masks, timing, and shared reward."""

    def __init__(
        self,
        wrapper,
        policies: dict[str, IndependentPolicy],
        builders: dict[str, CC4ModelInputBuilder],
    ):
        expected = tuple(wrapper.possible_agents)
        if set(policies) != set(expected) or set(builders) != set(expected):
            raise ValueError("A policy and input builder are required for each Blue agent")
        if len({id(policy) for policy in policies.values()}) != len(policies):
            raise ValueError("Policies must be independently instantiated per agent")

        self.wrapper = wrapper
        self.policies = policies
        self.builders = builders
        self.action_specs = {
            agent: policy.action_spec for agent, policy in policies.items()
        }
        self.sleep_indices: dict[str, int] = {}
        for agent, spec in self.action_specs.items():
            if len(spec.labels) != wrapper.action_space(agent).n:
                raise ValueError(f"Action mapping size mismatch for {agent}")
            sleep = [i for i, label in enumerate(spec.labels) if label == "Sleep"]
            if len(sleep) != 1:
                raise ValueError(f"Expected exactly one non-padding Sleep action for {agent}")
            self.sleep_indices[agent] = sleep[0]

    @staticmethod
    def _shared_reward(rewards: dict[str, float], agents: tuple[str, ...]) -> float:
        missing = [agent for agent in agents if agent not in rewards]
        if missing:
            raise RuntimeError(f"CC4 omitted Blue rewards for agents: {missing}")
        values = {agent: float(rewards[agent]) for agent in agents}
        first = values[agents[0]]
        if not all(math.isclose(value, first, rel_tol=0.0, abs_tol=1e-8) for value in values.values()):
            raise RuntimeError(
                "Blue rewards are not identical team-reward copies; refusing to "
                f"aggregate them as one shared reward: {values}"
            )
        return first

    def run(
        self,
        max_environment_steps: int = 8,
        *,
        seed: int | None = 21,
        deterministic: bool = False,
    ) -> CC4RolloutResult:
        """Reset and run a bounded rollout with no message-vector input."""

        if max_environment_steps < 1:
            raise ValueError("max_environment_steps must be positive")

        observations, infos = self.wrapper.reset(seed=seed)
        for policy in self.policies.values():
            policy.eval()

        # Remaining simulator ticks for the action issued by each agent.
        pending_ticks = {agent: 0 for agent in self.policies}
        result = CC4RolloutResult(
            decisions_by_agent={agent: [] for agent in self.policies}
        )

        for environment_step in range(max_environment_steps):
            active_agents = tuple(agent for agent in self.wrapper.agents if agent in self.policies)
            if not active_agents:
                break

            action_indices: dict[str, int] = {}
            action_labels: dict[str, str] = {}
            decision_data: dict[str, tuple[CC4ModelInput, Tensor, float, int]] = {}
            decision_agents: list[str] = []
            waiting_agents: list[str] = []
            next_pending = dict(pending_ticks)

            with torch.no_grad():
                for agent in active_agents:
                    if agent not in observations or agent not in infos:
                        raise RuntimeError(f"Missing current observation or info for {agent}")
                    if "action_mask" not in infos[agent]:
                        raise RuntimeError(f"Missing action mask for {agent}")

                    mask = torch.as_tensor(infos[agent]["action_mask"], dtype=torch.bool)
                    spec = self.action_specs[agent]
                    if mask.numel() != spec.action_size:
                        raise RuntimeError(f"Action mask size mismatch for {agent}")
                    sleep_index = self.sleep_indices[agent]
                    if not bool(mask[sleep_index]):
                        raise RuntimeError(f"CC4 invariant failed: Sleep is masked for {agent}")

                    if pending_ticks[agent] > 0:
                        # The simulator ignores new commands while a prior
                        # multi-tick action is running. Send Sleep and do not
                        # record this as a policy decision.
                        action_index = sleep_index
                        next_pending[agent] = pending_ticks[agent] - 1
                        waiting_agents.append(agent)
                    else:
                        local_observation = self.builders[agent].adapter.parse(
                            observations[agent], infos[agent]["action_mask"]
                        )
                        try:
                            device = next(self.policies[agent].parameters()).device
                        except StopIteration:
                            device = torch.device("cpu")
                        model_input = self.builders[agent].build(
                            local_observation, device=device
                        )
                        logits = self.policies[agent](model_input)
                        if logits.shape != (1, spec.action_size):
                            raise RuntimeError(
                                f"Unexpected logits shape for {agent}: {tuple(logits.shape)}"
                            )

                        if deterministic:
                            action_index = spec.select_greedy(logits[0], mask)
                            distribution = spec.masked_distribution(logits, mask)
                            log_probability = distribution.log_prob(
                                torch.tensor([action_index], device=logits.device)
                            ).squeeze(0)
                        else:
                            action_index, log_probability = spec.sample_action(logits[0], mask)

                        if not bool(mask[action_index]):
                            raise RuntimeError(f"Policy selected masked action for {agent}")
                        action = self.wrapper.actions(agent)[action_index]
                        duration = int(getattr(action, "duration", 1))
                        if duration < 1:
                            raise RuntimeError(f"Invalid action duration for {agent}: {duration}")
                        next_pending[agent] = duration - 1
                        stored_input = CC4ModelInput(
                            model_input.candidate_features.detach().cpu().clone(),
                            model_input.context.detach().cpu().clone(),
                        )
                        decision_data[agent] = (
                            stored_input,
                            mask.detach().cpu().clone(),
                            float(log_probability.item()),
                            duration,
                        )
                        decision_agents.append(agent)

                    action_indices[agent] = action_index
                    action_labels[agent] = spec.label(action_index)

            # No messages are passed: BlueFixedActionWrapper supplies its
            # default empty payload, which is excluded by the adapter.
            next_observations, rewards, terminated, truncated, next_infos = self.wrapper.step(
                action_indices
            )
            team_reward = self._shared_reward(rewards, tuple(self.policies))
            result.team_return += team_reward
            result.steps.append(
                RolloutStep(
                    environment_step=environment_step,
                    active_agents=active_agents,
                    decisions=tuple(decision_agents),
                    forced_wait_agents=tuple(waiting_agents),
                    action_indices=action_indices.copy(),
                    action_labels=action_labels.copy(),
                    team_reward=team_reward,
                    per_agent_rewards={agent: float(rewards[agent]) for agent in self.policies},
                    terminated={agent: bool(terminated.get(agent, False)) for agent in self.policies},
                    truncated={agent: bool(truncated.get(agent, False)) for agent in self.policies},
                )
            )

            for agent in decision_agents:
                model_input, mask, log_probability, duration = decision_data[agent]
                result.decisions_by_agent[agent].append(
                    AgentDecision(
                        environment_step=environment_step,
                        agent_name=agent,
                        action_index=action_indices[agent],
                        action_label=action_labels[agent],
                        action_duration=duration,
                        old_log_probability=log_probability,
                        model_input=model_input,
                        action_mask=mask,
                    )
                )

            pending_ticks = next_pending
            observations, infos = next_observations, next_infos
            if not self.wrapper.agents:
                break

        return result


class CC4VariableRolloutRunner:
    """Run a short rollout for one of the variable-host representations."""

    def __init__(self, wrapper, policies, adapters, input_builder):
        expected = tuple(wrapper.possible_agents)
        if set(policies) != set(expected) or set(adapters) != set(expected):
            raise ValueError("A policy and variable adapter are required for each Blue agent")
        if len({id(policy) for policy in policies.values()}) != len(policies):
            raise ValueError("Policies must be independently instantiated per agent")
        self.wrapper = wrapper
        self.policies = policies
        self.adapters = adapters
        self.input_builder = input_builder
        self.action_specs = {
            agent: policy.action_spec for agent, policy in policies.items()
        }
        self.sleep_indices = {}
        for agent, spec in self.action_specs.items():
            sleep = [i for i, label in enumerate(spec.labels) if label == "Sleep"]
            if len(sleep) != 1:
                raise ValueError(f"Expected one non-padding Sleep action for {agent}")
            self.sleep_indices[agent] = sleep[0]

    def run(self, max_environment_steps: int = 4, *, seed: int | None = 21):
        if max_environment_steps < 1:
            raise ValueError("max_environment_steps must be positive")
        observations, infos = self.wrapper.reset(seed=seed)
        for policy in self.policies.values():
            policy.eval()
        pending_ticks = {agent: 0 for agent in self.policies}
        result = CC4RolloutResult(
            decisions_by_agent={agent: [] for agent in self.policies}
        )

        for environment_step in range(max_environment_steps):
            active_agents = tuple(
                agent for agent in self.wrapper.agents if agent in self.policies
            )
            if not active_agents:
                break
            action_indices = {}
            action_labels = {}
            decisions = []
            forced_waits = []
            next_pending = dict(pending_ticks)

            with torch.no_grad():
                for agent in active_agents:
                    mask = torch.as_tensor(infos[agent]["action_mask"], dtype=torch.bool)
                    spec = self.action_specs[agent]
                    if mask.numel() != spec.action_size:
                        raise RuntimeError(f"Action mask size mismatch for {agent}")
                    sleep_index = self.sleep_indices[agent]
                    if not bool(mask[sleep_index]):
                        raise RuntimeError(f"Sleep is masked for {agent}")

                    if pending_ticks[agent] > 0:
                        action_index = sleep_index
                        next_pending[agent] = pending_ticks[agent] - 1
                        forced_waits.append(agent)
                    else:
                        parsed = self.adapters[agent].parse(
                            observations[agent], infos[agent]["action_mask"]
                        )
                        try:
                            device = next(self.policies[agent].parameters()).device
                        except StopIteration:
                            device = torch.device("cpu")
                        model_input = self.input_builder.build(parsed, device=device)
                        batched_input = VariableCC4ModelInput(
                            model_input.host_features.unsqueeze(0),
                            model_input.host_subnet_indices.unsqueeze(0),
                            model_input.context.unsqueeze(0),
                        )
                        logits = self.policies[agent](batched_input)
                        action_index = spec.select_greedy(logits[0], mask)
                        if not bool(mask[action_index]):
                            raise RuntimeError(f"Policy selected masked action for {agent}")
                        action = self.wrapper.actions(agent)[action_index]
                        duration = int(getattr(action, "duration", 1))
                        next_pending[agent] = duration - 1
                        decisions.append(agent)
                        result.decisions_by_agent[agent].append(
                            AgentDecision(
                                environment_step=environment_step,
                                agent_name=agent,
                                action_index=action_index,
                                action_label=spec.label(action_index),
                                action_duration=duration,
                                old_log_probability=0.0,
                                model_input=VariableCC4ModelInput(
                                    model_input.host_features.detach().cpu().clone(),
                                    model_input.host_subnet_indices.detach().cpu().clone(),
                                    model_input.context.detach().cpu().clone(),
                                ),
                                action_mask=mask.clone(),
                            )
                        )

                    action_indices[agent] = action_index
                    action_labels[agent] = spec.label(action_index)

            next_observations, rewards, terminated, truncated, next_infos = self.wrapper.step(
                action_indices
            )
            team_reward = CC4RolloutRunner._shared_reward(
                rewards, tuple(self.policies)
            )
            result.team_return += team_reward
            result.steps.append(
                RolloutStep(
                    environment_step=environment_step,
                    active_agents=active_agents,
                    decisions=tuple(decisions),
                    forced_wait_agents=tuple(forced_waits),
                    action_indices=action_indices.copy(),
                    action_labels=action_labels.copy(),
                    team_reward=team_reward,
                    per_agent_rewards={
                        agent: float(rewards[agent]) for agent in self.policies
                    },
                    terminated={
                        agent: bool(terminated.get(agent, False))
                        for agent in self.policies
                    },
                    truncated={
                        agent: bool(truncated.get(agent, False))
                        for agent in self.policies
                    },
                )
            )
            pending_ticks = next_pending
            observations, infos = next_observations, next_infos
            if not self.wrapper.agents:
                break
        return result
