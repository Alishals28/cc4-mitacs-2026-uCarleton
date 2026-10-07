import numpy as np
import pytest

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4LocalObservation import (
    CC4LocalObservationAdapter,
    MESSAGE_PAYLOAD_SIZE,
)
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


@pytest.fixture
def blue_flat_env():
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        steps=3,
    )
    return BlueFlatWrapper(CybORG(scenario_generator=scenario, seed=21), pad_spaces=True)


def test_local_observation_shapes_and_message_omission(blue_flat_env):
    observations, infos = blue_flat_env.reset()

    assert len(blue_flat_env.agents) == 5
    for agent in blue_flat_env.agents:
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        n_subnets = len(blue_flat_env.subnets(agent))

        assert parsed.subnet_context.shape == (n_subnets, 27)
        assert parsed.host_events.shape == (n_subnets, 16, 2)
        assert parsed.action_mask.shape == (blue_flat_env.action_space(agent).n,)
        assert np.array_equal(parsed.action_mask, infos[agent]["action_mask"])

        # The payload is between the agent's local features and any wrapper padding.
        message_start = adapter.unpadded_size - MESSAGE_PAYLOAD_SIZE
        changed_messages = observations[agent].copy()
        changed_messages[message_start : adapter.unpadded_size] = 1
        parsed_with_changed_messages = adapter.parse(
            changed_messages, infos[agent]["action_mask"]
        )
        assert np.array_equal(parsed.subnet_context, parsed_with_changed_messages.subnet_context)
        assert np.array_equal(parsed.host_events, parsed_with_changed_messages.host_events)


def test_local_observation_rejects_nonzero_wrapper_padding(blue_flat_env):
    observations, infos = blue_flat_env.reset()
    agent = "blue_agent_0"
    adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
    corrupted = observations[agent].copy()
    corrupted[adapter.unpadded_size] = 1

    with pytest.raises(ValueError, match="padding"):
        adapter.parse(corrupted, infos[agent]["action_mask"])


def test_local_observation_rejects_malformed_length(blue_flat_env):
    observations, infos = blue_flat_env.reset()

    for agent in blue_flat_env.agents:
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        malformed = observations[agent][: adapter.unpadded_size - 1]

        with pytest.raises(ValueError, match="at least"):
            adapter.parse(malformed, infos[agent]["action_mask"])
