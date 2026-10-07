"""Action-mask-aware primitives shared by the CC4 PPO/DQN learners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from CybORG.Agents.CC4Representations import (
    AgentActionSpec,
    VariableCC4ModelInput,
    VariableIndependentPolicy,
)
from CybORG.Agents.CC4Rollout import CC4RolloutRunner


def masked_ppo_log_prob_and_entropy(
    action_spec: AgentActionSpec,
    logits: Tensor,
    action_masks: Tensor,
    actions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute PPO policy terms using only currently valid actions.

    This is a minibatch utility, not a complete PPO trainer. It rejects a
    stored action if that action was invalid under its stored action mask.
    """

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError("PPO logits must have shape [batch, actions]")

    batch_size = logits.shape[0]
    masks = torch.as_tensor(action_masks, dtype=torch.bool, device=logits.device)
    if masks.ndim == 1:
        masks = masks.unsqueeze(0).expand(batch_size, -1)
    if masks.shape != logits.shape:
        raise ValueError("PPO action masks must match logits or be one shared action mask")

    actions = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
    if actions.numel() != batch_size:
        raise ValueError("PPO needs exactly one selected action per batch item")
    actions = actions.reshape(batch_size)
    if torch.any(actions < 0) or torch.any(actions >= action_spec.action_size):
        raise ValueError("PPO action index is outside this agent's action space")
    if not torch.all(masks.gather(1, actions.unsqueeze(1))):
        raise ValueError("PPO batch contains an action invalid under its stored mask")

    distribution = action_spec.masked_distribution(logits, masks)
    return distribution.log_prob(actions), distribution.entropy()


def masked_dqn_targets(
    next_q_values: Tensor,
    next_action_masks: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    gamma: float,
) -> Tensor:
    """Build one-step DQN targets with invalid next actions excluded.

    ``terminated`` marks absorbing terminal states. A terminal row does not
    bootstrap and is allowed to have an all-false next-action mask. Every
    nonterminal row must have at least one valid next action. Callers decide
    explicitly whether environment truncation should also stop bootstrapping.
    """

    if next_q_values.ndim == 1:
        next_q_values = next_q_values.unsqueeze(0)
    if next_q_values.ndim != 2 or next_q_values.shape[-1] == 0:
        raise ValueError("DQN next Q-values must have shape [batch, actions]")
    if not next_q_values.is_floating_point():
        raise TypeError("DQN Q-values must be floating point")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("DQN discount gamma must be between zero and one")

    batch_size, action_count = next_q_values.shape
    masks = torch.as_tensor(
        next_action_masks, dtype=torch.bool, device=next_q_values.device
    )
    if masks.ndim == 1:
        masks = masks.unsqueeze(0).expand(batch_size, -1)
    if masks.shape != next_q_values.shape:
        raise ValueError("DQN next-action masks must match next Q-values")

    rewards = torch.as_tensor(rewards, dtype=next_q_values.dtype, device=next_q_values.device)
    terminals = torch.as_tensor(terminated, dtype=torch.bool, device=next_q_values.device)
    if rewards.numel() != batch_size or terminals.numel() != batch_size:
        raise ValueError("DQN rewards and terminal flags must have one value per batch row")
    rewards = rewards.reshape(batch_size)
    terminals = terminals.reshape(batch_size)
    if not torch.isfinite(rewards).all():
        raise ValueError("DQN rewards must be finite")

    nonterminal_without_action = ~masks.any(dim=-1) & ~terminals
    if torch.any(nonterminal_without_action):
        raise ValueError("A nonterminal DQN next state has no valid actions")
    if not torch.isfinite(next_q_values.masked_select(masks)).all():
        raise ValueError("Valid DQN next-action Q-values must be finite")

    masked_q = next_q_values.masked_fill(~masks, torch.finfo(next_q_values.dtype).min)
    max_next_q = masked_q.max(dim=-1).values
    max_next_q = torch.where(terminals, torch.zeros_like(max_next_q), max_next_q)
    return rewards + float(gamma) * max_next_q


@dataclass(frozen=True)
class PPOConfig:
    """Small, explicit PPO settings for independent variable-host policies."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 32

    def __post_init__(self):
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("PPO gamma must be between zero and one")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("PPO GAE lambda must be between zero and one")
        if self.clip_epsilon <= 0.0:
            raise ValueError("PPO clip epsilon must be positive")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("PPO learning rate and gradient norm must be positive")
        if self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("PPO epochs and minibatch size must be positive")


@dataclass
class PPOTransition:
    """One policy decision with discounted rewards across its action duration."""

    model_input: VariableCC4ModelInput
    action_mask: Tensor
    action: int
    old_log_probability: float
    value: float
    discounted_reward: float
    duration: int
    next_value: float
    terminated: bool


@dataclass
class _OpenPPODecision:
    model_input: VariableCC4ModelInput
    action_mask: Tensor
    action: int
    old_log_probability: float
    value: float
    discounted_reward: float = 0.0
    elapsed_ticks: int = 0


def semi_markov_gae(
    transitions: Sequence[PPOTransition],
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE where each PPO decision can span multiple env ticks.

    Each transition's reward is already the discounted sum of all shared team
    rewards observed during that action and its forced waits. Both bootstrap
    discount and GAE trace decay therefore use the elapsed duration.
    """

    advantages = torch.zeros(len(transitions), dtype=torch.float32)
    values = torch.tensor([item.value for item in transitions], dtype=torch.float32)
    gae = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        discount = float(gamma) ** item.duration
        continuation = 0.0 if item.terminated else 1.0
        delta = (
            item.discounted_reward
            + discount * continuation * item.next_value
            - item.value
        )
        gae = delta + discount * (float(gae_lambda) ** item.duration) * continuation * gae
        advantages[index] = gae
    return advantages, advantages + values


class CC4VariablePPOTrainer:
    """Independent masked PPO learners for the three variable-host encoders.

    One environment episode supplies one shared Blue reward per tick. Each
    independent agent receives its own copy as its learning signal; rewards
    are never summed across the five copies. Multi-tick actions produce one
    semi-Markov transition whose reward includes every tick until the agent's
    next policy decision.
    """

    def __init__(
        self,
        wrapper,
        policies: dict[str, VariableIndependentPolicy],
        adapters: dict,
        input_builder,
        config: PPOConfig | None = None,
    ):
        expected = tuple(wrapper.possible_agents)
        if set(policies) != set(expected) or set(adapters) != set(expected):
            raise ValueError("PPO requires one policy and adapter per Blue agent")
        if len({id(policy) for policy in policies.values()}) != len(policies):
            raise ValueError("PPO policies must have independent parameters")
        self.wrapper = wrapper
        self.policies = policies
        self.adapters = adapters
        self.input_builder = input_builder
        self.config = config or PPOConfig()
        self.optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)
            for agent, policy in policies.items()
        }
        self.optimizer_updates = {agent: 0 for agent in policies}
        self.last_episode_team_return = 0.0
        self.last_episode_environment_steps = 0
        self.action_specs = {agent: policy.action_spec for agent, policy in policies.items()}
        self.sleep_indices = {}
        for agent, spec in self.action_specs.items():
            if len(spec.labels) != wrapper.action_space(agent).n:
                raise ValueError(f"Action mapping mismatch for {agent}")
            sleep = [index for index, label in enumerate(spec.labels) if label == "Sleep"]
            if len(sleep) != 1:
                raise ValueError(f"Expected exactly one Sleep action for {agent}")
            self.sleep_indices[agent] = sleep[0]

    @staticmethod
    def _cpu_input(model_input: VariableCC4ModelInput) -> VariableCC4ModelInput:
        return VariableCC4ModelInput(
            host_features=model_input.host_features.detach().cpu().clone(),
            host_subnet_indices=model_input.host_subnet_indices.detach().cpu().clone(),
            context=model_input.context.detach().cpu().clone(),
        )

    @staticmethod
    def _batched_input(
        inputs: Sequence[VariableCC4ModelInput], device: torch.device
    ) -> VariableCC4ModelInput:
        if not inputs:
            raise ValueError("Cannot batch an empty PPO input list")
        subnet_shape = inputs[0].context.shape
        if any(item.context.shape != subnet_shape for item in inputs):
            raise ValueError("A minibatch must use one agent's fixed local subnet layout")
        batch_size = len(inputs)
        max_hosts = max(item.host_features.shape[0] for item in inputs)
        host_features = torch.zeros(
            (batch_size, max_hosts, 4), dtype=torch.float32, device=device
        )
        host_indices = torch.zeros(
            (batch_size, max_hosts), dtype=torch.long, device=device
        )
        padding_mask = torch.ones(
            (batch_size, max_hosts), dtype=torch.bool, device=device
        )
        contexts = torch.stack([item.context for item in inputs]).to(device)
        for row, item in enumerate(inputs):
            count = item.host_features.shape[0]
            if count:
                host_features[row, :count] = item.host_features.to(device)
                host_indices[row, :count] = item.host_subnet_indices.to(device)
                padding_mask[row, :count] = False
        return VariableCC4ModelInput(
            host_features=host_features,
            host_subnet_indices=host_indices,
            context=contexts,
            host_padding_mask=padding_mask,
        )

    @torch.no_grad()
    def collect_episode(self, seed: int | None = None) -> dict[str, list[PPOTransition]]:
        """Collect a complete episode, preserving tick rewards and action timing."""

        observations, infos = self.wrapper.reset(seed=seed)
        pending_ticks = {agent: 0 for agent in self.policies}
        open_decisions: dict[str, _OpenPPODecision | None] = {
            agent: None for agent in self.policies
        }
        transitions = {agent: [] for agent in self.policies}
        episode_team_return = 0.0
        episode_environment_steps = 0
        for policy in self.policies.values():
            policy.eval()

        while self.wrapper.agents:
            active_agents = tuple(
                agent for agent in self.wrapper.agents if agent in self.policies
            )
            if not active_agents:
                break
            actions: dict[str, int] = {}
            decision_agents: set[str] = set()
            forced_wait_agents: set[str] = set()
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
                    raise RuntimeError(f"Sleep is masked for {agent}")

                if pending_ticks[agent] > 0:
                    actions[agent] = sleep_index
                    forced_wait_agents.add(agent)
                    continue

                parsed = self.adapters[agent].parse(
                    observations[agent], infos[agent]["action_mask"]
                )
                try:
                    device = next(self.policies[agent].parameters()).device
                except StopIteration:
                    device = torch.device("cpu")
                model_input = self.input_builder.build(parsed, device=device)
                logits, values = self.policies[agent].evaluate(model_input)
                action, log_probability = spec.sample_action(logits[0], mask)
                if not bool(mask[action]):
                    raise RuntimeError(f"PPO sampled an invalid action for {agent}")

                if open_decisions[agent] is not None:
                    previous = open_decisions[agent]
                    transitions[agent].append(
                        PPOTransition(
                            model_input=previous.model_input,
                            action_mask=previous.action_mask,
                            action=previous.action,
                            old_log_probability=previous.old_log_probability,
                            value=previous.value,
                            discounted_reward=previous.discounted_reward,
                            duration=previous.elapsed_ticks,
                            next_value=float(values[0].item()),
                            terminated=False,
                        )
                    )

                action_obj = self.wrapper.actions(agent)[action]
                duration = int(getattr(action_obj, "duration", 1))
                if duration < 1:
                    raise RuntimeError(f"Invalid action duration {duration} for {agent}")
                open_decisions[agent] = _OpenPPODecision(
                    model_input=self._cpu_input(model_input),
                    action_mask=mask.clone(),
                    action=action,
                    old_log_probability=float(log_probability.item()),
                    value=float(values[0].item()),
                )
                pending_ticks[agent] = duration - 1
                actions[agent] = action
                decision_agents.add(agent)

            next_observations, rewards, terminated, truncated, next_infos = self.wrapper.step(
                actions
            )
            team_reward = CC4RolloutRunner._shared_reward(
                rewards, tuple(self.policies)
            )
            episode_team_return += team_reward
            episode_environment_steps += 1
            for agent, opened in open_decisions.items():
                if opened is not None:
                    opened.discounted_reward += (
                        self.config.gamma ** opened.elapsed_ticks
                    ) * team_reward
                    opened.elapsed_ticks += 1
            for agent in forced_wait_agents:
                pending_ticks[agent] -= 1

            episode_done = (
                not self.wrapper.agents
                or any(terminated.get(agent, False) for agent in self.policies)
                or any(truncated.get(agent, False) for agent in self.policies)
            )
            if episode_done:
                for agent, opened in open_decisions.items():
                    if opened is not None:
                        transitions[agent].append(
                            PPOTransition(
                                model_input=opened.model_input,
                                action_mask=opened.action_mask,
                                action=opened.action,
                                old_log_probability=opened.old_log_probability,
                                value=opened.value,
                                discounted_reward=opened.discounted_reward,
                                duration=max(1, opened.elapsed_ticks),
                                next_value=0.0,
                                terminated=True,
                            )
                        )
                        open_decisions[agent] = None

            observations, infos = next_observations, next_infos

        self.last_episode_team_return = episode_team_return
        self.last_episode_environment_steps = episode_environment_steps
        return transitions

    def update(self, trajectories: dict[str, list[PPOTransition]]) -> dict[str, float]:
        """Perform masked clipped-PPO updates with one optimizer per agent."""

        metrics: dict[str, float] = {}
        for agent, policy in self.policies.items():
            items = trajectories.get(agent, [])
            if not items:
                raise ValueError(f"No PPO decisions collected for {agent}")
            advantages, returns = semi_markov_gae(
                items, self.config.gamma, self.config.gae_lambda
            )
            if len(items) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )
            device = next(policy.parameters()).device
            old_log_probs = torch.tensor(
                [item.old_log_probability for item in items],
                dtype=torch.float32,
                device=device,
            )
            actions = torch.tensor([item.action for item in items], device=device)
            masks = torch.stack([item.action_mask for item in items]).to(device)
            advantages = advantages.to(device)
            returns = returns.to(device)
            optimizer = self.optimizers[agent]
            accumulated_loss = 0.0
            update_count = 0
            policy.train()

            for _ in range(self.config.epochs):
                order = torch.randperm(len(items))
                for start in range(0, len(items), self.config.minibatch_size):
                    indices = order[start : start + self.config.minibatch_size]
                    batch_items = [items[int(index)] for index in indices]
                    model_input = self._batched_input(
                        [item.model_input for item in batch_items], device
                    )
                    logits, values = policy.evaluate(model_input)
                    batch_actions = actions[indices.to(device)]
                    batch_masks = masks[indices.to(device)]
                    new_log_probs, entropy = masked_ppo_log_prob_and_entropy(
                        policy.action_spec,
                        logits,
                        batch_masks,
                        batch_actions,
                    )
                    ratio = torch.exp(new_log_probs - old_log_probs[indices.to(device)])
                    batch_advantages = advantages[indices.to(device)]
                    unclipped = ratio * batch_advantages
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * batch_advantages
                    actor_loss = -torch.minimum(unclipped, clipped).mean()
                    critic_loss = 0.5 * (values - returns[indices.to(device)]).pow(2).mean()
                    loss = (
                        actor_loss
                        + self.config.value_coefficient * critic_loss
                        - self.config.entropy_coefficient * entropy.mean()
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip_grad_norm_(policy.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    self.optimizer_updates[agent] += 1
                    update_count += 1
                    accumulated_loss += float(loss.detach().item())

            policy.eval()
            metrics[f"{agent}/loss"] = accumulated_loss / update_count
            metrics[f"{agent}/transitions"] = float(len(items))
            metrics[f"{agent}/optimizer_updates"] = float(update_count)
        return metrics

    def train_episode(self, seed: int | None = None) -> dict[str, float]:
        """Collect one complete episode and update each independent policy."""

        trajectories = self.collect_episode(seed=seed)
        metrics = self.update(trajectories)
        metrics["episode/team_return"] = self.last_episode_team_return
        metrics["episode/environment_steps"] = float(
            self.last_episode_environment_steps
        )
        metrics["episode/decisions"] = float(
            sum(len(items) for items in trajectories.values())
        )
        return metrics
