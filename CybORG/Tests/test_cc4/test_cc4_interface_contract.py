import numpy as np
import pytest

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4LocalObservation import CC4LocalObservationAdapter
from CybORG.Agents.CC4Representations import AgentActionSpec
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


@pytest.fixture
def interface_env():
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        steps=9,
    )
    return BlueFlatWrapper(
        CybORG(scenario_generator=scenario, seed=21),
        pad_spaces=True,
    )


def _sleep_actions(wrapper):
    return {
        agent: wrapper.action_labels(agent).index("Sleep")
        for agent in wrapper.possible_agents
    }


def test_interface_contract_covers_agents_phases_and_shared_rewards(interface_env):
    observations, infos = interface_env.reset(seed=21)
    expected_agents = tuple(interface_env.possible_agents)
    observed_phases = set()
    previous_reward_values = None

    for _ in range(9):
        for agent in expected_agents:
            adapter = CC4LocalObservationAdapter(interface_env, agent)
            parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
            spec = AgentActionSpec.from_wrapper(interface_env, agent)
            observed_phases.add(parsed.mission_phase)

            assert parsed.action_mask.shape == (spec.action_size,)
            assert len(spec.labels) == interface_env.action_space(agent).n
            if agent == "blue_agent_4":
                assert parsed.subnet_context.shape == (3, 27)
                assert parsed.host_events.shape == (3, 16, 2)
            else:
                assert parsed.subnet_context.shape == (1, 27)
                assert parsed.host_events.shape == (1, 16, 2)

        observations, rewards, _, _, infos = interface_env.step(
            _sleep_actions(interface_env)
        )
        reward_values = [float(rewards[agent]) for agent in expected_agents]
        if previous_reward_values is not None:
            assert len(reward_values) == len(previous_reward_values)
        assert np.allclose(reward_values, reward_values[0])
        previous_reward_values = reward_values

    assert observed_phases == {0, 1, 2}


def test_duration_two_action_waits_before_execution(interface_env):
    observations, infos = interface_env.reset(seed=21)
    agent = "blue_agent_0"
    valid_duration_two = next(
        index
        for index, (action, valid) in enumerate(
            zip(interface_env.actions(agent), infos[agent]["action_mask"])
        )
        if valid and action.__class__.__name__ != "Sleep" and action.duration == 2
    )
    sleep_actions = _sleep_actions(interface_env)
    selected_label = interface_env.action_labels(agent)[valid_duration_two]
    first_actions = dict(sleep_actions)
    first_actions[agent] = valid_duration_two

    interface_env.step(first_actions)
    first_execution = interface_env.env.environment_controller.action[agent][0]
    assert first_execution.__class__.__name__ == "Sleep"

    interface_env.step(sleep_actions)
    second_execution = interface_env.env.environment_controller.action[agent][0]
    assert second_execution.__class__.__name__ == selected_label.split()[0]
