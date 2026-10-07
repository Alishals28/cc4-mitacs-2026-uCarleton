import numpy as np
import pytest
import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4LocalObservation import CC4LocalObservationAdapter
from CybORG.Agents.CC4Representations import CC4ModelInputBuilder
from CybORG.Agents.CC4Rollout import CC4RolloutRunner, build_independent_policies
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


REPRESENTATIONS = ("zero_padding", "deep_sets")


@pytest.fixture
def baseline_env():
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        steps=6,
    )
    return BlueFlatWrapper(
        CybORG(scenario_generator=scenario, seed=21),
        pad_spaces=True,
    )


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_baseline_policies_are_independent_and_use_common_inputs(
    baseline_env, representation
):
    observations, infos = baseline_env.reset(seed=21)
    policies, builders = build_independent_policies(baseline_env, representation)

    assert set(policies) == set(baseline_env.possible_agents)
    assert len({id(policy) for policy in policies.values()}) == 5

    parameter_ids = [
        id(parameter)
        for policy in policies.values()
        for parameter in policy.parameters()
    ]
    assert len(parameter_ids) == len(set(parameter_ids))

    for agent in baseline_env.possible_agents:
        adapter = CC4LocalObservationAdapter(baseline_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        expected_input = CC4ModelInputBuilder(adapter).build(parsed)
        actual_input = builders[agent].build(parsed)

        assert torch.equal(expected_input.candidate_features, actual_input.candidate_features)
        assert torch.equal(expected_input.context, actual_input.context)

        logits = policies[agent](
            type(actual_input)(
                actual_input.candidate_features.unsqueeze(0),
                actual_input.context.unsqueeze(0),
            )
        )
        assert logits.shape == (1, baseline_env.action_space(agent).n)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_each_baseline_completes_a_masked_shared_reward_rollout(
    baseline_env, representation
):
    torch.manual_seed(123)
    baseline_env.reset(seed=21)
    policies, builders = build_independent_policies(baseline_env, representation)
    result = CC4RolloutRunner(baseline_env, policies, builders).run(
        max_environment_steps=4,
        seed=21,
        deterministic=True,
    )

    assert len(result.steps) == 4
    assert set(result.decisions_by_agent) == set(baseline_env.possible_agents)
    assert all(result.decisions_by_agent[agent] for agent in baseline_env.possible_agents)
    assert np.isclose(result.team_return, sum(step.team_reward for step in result.steps))

    for step in result.steps:
        assert np.allclose(
            list(step.per_agent_rewards.values()),
            step.team_reward,
        )
        for agent, action_index in step.action_indices.items():
            assert step.action_labels[agent] == baseline_env.action_labels(agent)[action_index]
            assert action_index >= 0
            assert action_index < baseline_env.action_space(agent).n

    for decisions in result.decisions_by_agent.values():
        for decision in decisions:
            assert bool(decision.action_mask[decision.action_index])
