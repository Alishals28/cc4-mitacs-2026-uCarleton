import pytest

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


AUDIT_SEEDS = tuple(range(50))


def _host_target(label: str, host_names: set[str]) -> str | None:
    """Extract a target only for host-target action labels, for test oracle use."""

    matches = [hostname for hostname in host_names if label.endswith(f" {hostname}")]
    if len(matches) == 1:
        return matches[0]
    return None


def _audit_wrapper(seed: int) -> BlueFlatWrapper:
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        steps=6,
    )
    wrapper = BlueFlatWrapper(
        CybORG(scenario_generator=scenario, seed=seed),
        pad_spaces=True,
    )
    wrapper.reset(seed=seed)
    return wrapper


@pytest.mark.parametrize("seed", AUDIT_SEEDS)
def test_action_mask_host_targets_match_presence_and_session_oracles(seed):
    """Audit public mask semantics against test-only simulator oracles.

    The raw simulator state is deliberately used only in this test. Policy
    adapters receive wrapper observations and action masks, never these sets.
    """

    wrapper = _audit_wrapper(seed)
    state = wrapper.env.environment_controller.state

    for agent in wrapper.possible_agents:
        wrapper_hosts = {
            hostname
            for hostname in wrapper.hosts(agent)
            if "router" not in hostname
        }
        actual_hosts = wrapper_hosts.intersection(state.hosts)
        session_hosts = {
            hostname
            for hostname in actual_hosts
            if state.hosts[hostname].sessions.get(agent)
        }

        masked_action_targets = set()
        for label, valid in zip(
            wrapper.action_labels(agent), wrapper.action_mask(agent)
        ):
            target = _host_target(label, wrapper_hosts)
            if target is not None and valid:
                masked_action_targets.add(target)

        details = {
            "seed": seed,
            "agent": agent,
            "mask_actionable_hosts": sorted(masked_action_targets),
            "physical_hosts": sorted(actual_hosts),
            "blue_session_hosts": sorted(session_hosts),
        }
        # BlueFixedActionWrapper masks a host-target action when the host is
        # absent or the agent has no session. Check these separately so the
        # test reveals whether the two conditions differ in sampled episodes.
        assert masked_action_targets == session_hosts, details
        assert masked_action_targets == actual_hosts, details
