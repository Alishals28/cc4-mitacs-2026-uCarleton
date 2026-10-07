import numpy as np
import pytest
import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4Representations import AgentActionSpec
from CybORG.Agents.CC4Rollout import CC4RolloutRunner, build_independent_policies
from CybORG.Agents.CC4Training import masked_dqn_targets, masked_ppo_log_prob_and_entropy
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


def test_masked_policy_and_ppo_terms_exclude_invalid_actions():
    spec = AgentActionSpec("blue_agent_test", ("Sleep", "invalid", "Analyse"))
    logits = torch.tensor([[0.0, 1000.0, 1.0]])
    mask = torch.tensor([[True, False, True]])
    distribution = spec.masked_distribution(logits, mask)

    assert distribution.probs[0, 1].item() == 0.0
    action, log_probability = spec.sample_action(logits[0], mask[0])
    assert action in (0, 2)
    assert torch.isfinite(log_probability)
    assert spec.sample_uniform_valid_action(mask[0]) in (0, 2)
    assert spec.select_epsilon_greedy(logits[0], mask[0], epsilon=1.0) in (0, 2)

    log_probs, entropy = masked_ppo_log_prob_and_entropy(
        spec, logits, mask, torch.tensor([2])
    )
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()
    with pytest.raises(ValueError, match="invalid under its stored mask"):
        masked_ppo_log_prob_and_entropy(spec, logits, mask, torch.tensor([1]))
    with pytest.raises(ValueError, match="All actions are masked"):
        spec.masked_distribution(logits, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="All actions are masked"):
        spec.sample_uniform_valid_action(torch.zeros(3, dtype=torch.bool))


def test_dqn_targets_mask_next_actions_and_handle_terminal_empty_mask():
    q_values = torch.tensor([[100.0, 4.0, 7.0], [3.0, 2.0, 1.0]])
    masks = torch.tensor([[False, True, False], [False, False, False]])
    targets = masked_dqn_targets(
        q_values,
        masks,
        rewards=torch.tensor([1.0, 2.0]),
        terminated=torch.tensor([False, True]),
        gamma=0.9,
    )
    assert torch.allclose(targets, torch.tensor([4.6, 2.0]))

    with pytest.raises(ValueError, match="nonterminal.*no valid actions"):
        masked_dqn_targets(
            q_values[:1], masks[:1] & False,
            rewards=torch.tensor([0.0]), terminated=torch.tensor([False]), gamma=0.9
        )


def test_five_independent_policies_complete_short_cc4_rollout():
    torch.manual_seed(123)
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        steps=6,
    )
    wrapper = BlueFlatWrapper(CybORG(scenario_generator=scenario, seed=21), pad_spaces=True)
    # Construct policies after one reset so wrapper action metadata is populated.
    wrapper.reset(seed=21)
    policies, builders = build_independent_policies(wrapper, "zero_padding")
    assert len({id(policy) for policy in policies.values()}) == 5
    runner = CC4RolloutRunner(wrapper, policies, builders)

    result = runner.run(max_environment_steps=4, seed=21)

    assert 1 <= len(result.steps) <= 4
    assert set(result.decisions_by_agent) == set(wrapper.possible_agents)
    assert all(result.decisions_by_agent[agent] for agent in wrapper.possible_agents)
    assert np.isclose(result.team_return, sum(step.team_reward for step in result.steps))
    for step in result.steps:
        assert set(step.per_agent_rewards) == set(wrapper.possible_agents)
        assert all(
            np.isclose(reward, step.team_reward)
            for reward in step.per_agent_rewards.values()
        )
        assert all(agent in step.active_agents for agent in step.decisions)
        assert all(
            step.action_labels[agent] == "Sleep"
            for agent in step.forced_wait_agents
        )
    for agent, decisions in result.decisions_by_agent.items():
        labels = wrapper.action_labels(agent)
        for decision in decisions:
            assert decision.action_label == labels[decision.action_index]
            assert decision.action_mask[decision.action_index]
            assert decision.agent_name == agent
