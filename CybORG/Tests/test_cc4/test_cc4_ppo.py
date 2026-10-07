import pytest
import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4Representations import VariableCC4ModelInput
from CybORG.Agents.CC4Rollout import build_variable_independent_policies
from CybORG.Agents.CC4Training import (
    CC4VariablePPOTrainer,
    PPOConfig,
    PPOTransition,
    semi_markov_gae,
)
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


def test_semi_markov_gae_discounts_tick_rewards_across_action_duration():
    empty_input = VariableCC4ModelInput(
        host_features=torch.empty((0, 4)),
        host_subnet_indices=torch.empty((0,), dtype=torch.long),
        context=torch.zeros((1, 30)),
    )
    transitions = [
        PPOTransition(
            model_input=empty_input,
            action_mask=torch.tensor([True]),
            action=0,
            old_log_probability=0.0,
            value=0.5,
            discounted_reward=2.8,  # 1.0 + 0.9 * 2.0 over two ticks
            duration=2,
            next_value=0.3,
            terminated=False,
        ),
        PPOTransition(
            model_input=empty_input,
            action_mask=torch.tensor([True]),
            action=0,
            old_log_probability=0.0,
            value=0.3,
            discounted_reward=4.0,
            duration=1,
            next_value=0.0,
            terminated=True,
        ),
    ]

    advantages, returns = semi_markov_gae(transitions, gamma=0.9, gae_lambda=0.95)

    delta_1 = 2.8 + (0.9**2) * 0.3 - 0.5
    delta_2 = 4.0 - 0.3
    expected_gae_1 = delta_1 + (0.9**2) * (0.95**2) * delta_2
    assert torch.allclose(advantages, torch.tensor([expected_gae_1, delta_2]))
    assert torch.allclose(returns, advantages + torch.tensor([0.5, 0.3]))


@pytest.mark.parametrize("representation", ("zero_padding", "deep_sets", "transformer"))
def test_variable_ppo_performs_optimizer_updates_for_all_five_agents(representation):
    torch.manual_seed(123)
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        steps=6,
    )
    wrapper = BlueFlatWrapper(
        CybORG(scenario_generator=scenario, seed=21), pad_spaces=True
    )
    wrapper.reset(seed=21)
    policies, adapters, input_builder = build_variable_independent_policies(
        wrapper, representation
    )
    trainer = CC4VariablePPOTrainer(
        wrapper,
        policies,
        adapters,
        input_builder,
        PPOConfig(epochs=1, minibatch_size=2),
    )
    before = {
        agent: {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
        }
        for agent, policy in policies.items()
    }

    metrics = trainer.train_episode(seed=21)

    assert set(metrics) >= {
        f"{agent}/optimizer_updates" for agent in wrapper.possible_agents
    }
    for agent, policy in policies.items():
        assert metrics[f"{agent}/transitions"] > 0
        assert metrics[f"{agent}/optimizer_updates"] >= 1
        assert trainer.optimizer_updates[agent] >= 1
        assert any(
            not torch.equal(value, before[agent][name])
            for name, value in policy.state_dict().items()
        )


def test_ppo_update_masks_invalid_actions_and_batched_host_padding():
    torch.manual_seed(5)
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        steps=6,
    )
    wrapper = BlueFlatWrapper(
        CybORG(scenario_generator=scenario, seed=21), pad_spaces=True
    )
    wrapper.reset(seed=21)
    policies, adapters, builder = build_variable_independent_policies(
        wrapper, "deep_sets"
    )
    trainer = CC4VariablePPOTrainer(
        wrapper, policies, adapters, builder, PPOConfig(epochs=1, minibatch_size=2)
    )
    trajectories = trainer.collect_episode(seed=21)

    for agent, items in trajectories.items():
        assert items
        assert all(item.action_mask[item.action] for item in items)
        assert all(item.duration >= 1 for item in items)
    metrics = trainer.update(trajectories)
    assert all(metrics[f"{agent}/optimizer_updates"] >= 1 for agent in policies)
