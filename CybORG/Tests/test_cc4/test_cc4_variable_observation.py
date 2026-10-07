import numpy as np
import pytest
import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4LocalObservation import MESSAGE_PAYLOAD_SIZE
from CybORG.Agents.CC4Representations import (
    AgentActionSpec,
    VariableCC4DeepSetsEncoder,
    VariableCC4InputBuilder,
    VariableCC4ModelInput,
    VariableCC4TransformerEncoder,
    VariableCC4ZeroPaddingEncoder,
    VariableIndependentPolicy,
)
from CybORG.Agents.CC4Rollout import (
    CC4VariableRolloutRunner,
    build_variable_independent_policies,
)
from CybORG.Agents.CC4VariableObservation import (
    CC4VariableHostAdapter,
    VARIABLE_HOST_FEATURES,
)
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


VARIABLE_SEEDS = (0, 4, 12, 19)


@pytest.fixture
def variable_env():
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


def _environment(seed):
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        steps=3,
    )
    wrapper = BlueFlatWrapper(
        CybORG(scenario_generator=scenario, seed=seed),
        pad_spaces=True,
    )
    return wrapper, wrapper.reset(seed=seed)


@pytest.mark.parametrize("seed", VARIABLE_SEEDS)
def test_variable_adapter_returns_actionable_tokens_for_all_agents(seed):
    wrapper, (observations, infos) = _environment(seed)

    for agent in wrapper.possible_agents:
        adapter = CC4VariableHostAdapter(wrapper, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])

        assert parsed.host_features.shape == (
            len(parsed.actionable_hosts),
            VARIABLE_HOST_FEATURES,
        )
        assert parsed.host_subnet_indices.shape == (len(parsed.actionable_hosts),)
        assert len(set(parsed.actionable_hosts)) == len(parsed.actionable_hosts)
        assert parsed.subnet_context.shape == (len(adapter.subnets), 27)
        assert parsed.action_mask.shape == (wrapper.action_space(agent).n,)
        assert parsed.action_labels == tuple(wrapper.action_labels(agent))
        assert np.array_equal(parsed.action_mask, infos[agent]["action_mask"])
        assert np.all(parsed.host_subnet_indices >= 0)
        assert np.all(parsed.host_subnet_indices < len(adapter.subnets))

        expected_hosts = tuple(
            hostname
            for subnet_hosts in adapter.host_names_by_subnet
            for hostname in subnet_hosts
            if any(
                valid and label.endswith(f" {hostname}")
                for label, valid in zip(parsed.action_labels, parsed.action_mask)
            )
        )
        assert parsed.actionable_hosts == expected_hosts

        for index, hostname in enumerate(parsed.actionable_hosts):
            assert any(
                valid and label.endswith(f" {hostname}")
                for label, valid in zip(parsed.action_labels, parsed.action_mask)
            )
            if "_user_host_" in hostname:
                assert np.array_equal(parsed.host_features[index, 2:], [1.0, 0.0])
            else:
                assert "_server_host_" in hostname
                assert np.array_equal(parsed.host_features[index, 2:], [0.0, 1.0])


def test_variable_adapter_preserves_one_and_three_subnet_layouts(variable_env):
    observations, infos = variable_env.reset(seed=21)

    for agent in variable_env.possible_agents:
        adapter = CC4VariableHostAdapter(variable_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        expected_subnets = 3 if agent == "blue_agent_4" else 1
        assert len(adapter.subnets) == expected_subnets
        assert parsed.subnet_context.shape == (expected_subnets, 27)
        assert np.all(parsed.host_subnet_indices < expected_subnets)


def test_variable_adapter_excludes_message_payload(variable_env):
    observations, infos = variable_env.reset(seed=21)
    agent = "blue_agent_4"
    adapter = CC4VariableHostAdapter(variable_env, agent)
    original = adapter.parse(observations[agent], infos[agent]["action_mask"])

    changed = observations[agent].copy()
    message_start = adapter.fixed_adapter.unpadded_size - MESSAGE_PAYLOAD_SIZE
    changed[message_start : adapter.fixed_adapter.unpadded_size] = 1
    parsed_changed = adapter.parse(changed, infos[agent]["action_mask"])

    assert np.array_equal(original.host_features, parsed_changed.host_features)
    assert np.array_equal(original.subnet_context, parsed_changed.subnet_context)
    assert original.actionable_hosts == parsed_changed.actionable_hosts


def test_variable_token_counts_differ_across_tested_scenarios():
    counts = set()

    for seed in VARIABLE_SEEDS:
        wrapper, (observations, infos) = _environment(seed)
        for agent in wrapper.possible_agents:
            adapter = CC4VariableHostAdapter(wrapper, agent)
            parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
            counts.add((agent, len(parsed.actionable_hosts)))

    assert len({count for _, count in counts}) > 1


def test_sampled_scenarios_stay_within_documented_legal_host_bounds():
    for seed in VARIABLE_SEEDS:
        wrapper, _ = _environment(seed)
        state = wrapper.env.environment_controller.state
        user_counts = {}
        server_counts = {}
        for hostname in state.hosts:
            if "_user_host_" in hostname:
                subnet_name = hostname.split("_user_host_", 1)[0]
                user_counts[subnet_name] = user_counts.get(subnet_name, 0) + 1
            if "_server_host_" in hostname:
                subnet_name = hostname.split("_server_host_", 1)[0]
                server_counts[subnet_name] = server_counts.get(subnet_name, 0) + 1

        assert user_counts
        assert server_counts
        assert all(3 <= count <= 10 for count in user_counts.values())
        assert all(1 <= count <= 6 for count in server_counts.values())


def _variable_input(wrapper, observations, infos, agent):
    parsed = CC4VariableHostAdapter(wrapper, agent).parse(
        observations[agent], infos[agent]["action_mask"]
    )
    model_input = VariableCC4InputBuilder().build(parsed)
    return VariableCC4ModelInput(
        host_features=model_input.host_features.unsqueeze(0),
        host_subnet_indices=model_input.host_subnet_indices.unsqueeze(0),
        context=model_input.context.unsqueeze(0),
    )


def test_variable_transformer_reuses_weights_for_different_host_counts():
    inputs = []
    counts = []
    for seed in VARIABLE_SEEDS:
        wrapper, (observations, infos) = _environment(seed)
        model_input = _variable_input(wrapper, observations, infos, "blue_agent_0")
        inputs.append(model_input)
        counts.append(model_input.host_features.shape[1])

    low_index = counts.index(min(counts))
    high_index = counts.index(max(counts))
    assert counts[low_index] != counts[high_index]

    encoder = VariableCC4TransformerEncoder().eval()
    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in encoder.state_dict().items()
    }
    assert not hasattr(encoder, "slot_embedding")

    outputs = [
        encoder(
            model_input.host_features,
            model_input.host_subnet_indices,
            model_input.context,
        )
        for model_input in (inputs[low_index], inputs[high_index])
    ]
    assert all(output.shape == (1, 128) for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)
    for name, parameter in encoder.state_dict().items():
        assert torch.equal(parameter, initial_state[name])


def test_variable_transformer_supports_empty_tokens_and_padding_invariance(variable_env):
    observations, infos = variable_env.reset(seed=21)
    model_input = _variable_input(variable_env, observations, infos, "blue_agent_0")
    encoder = VariableCC4TransformerEncoder().eval()

    empty_features = torch.empty((1, 0, 4), dtype=torch.float32)
    empty_indices = torch.empty((1, 0), dtype=torch.long)
    empty_output = encoder(empty_features, empty_indices, model_input.context)
    assert empty_output.shape == (1, 128)
    assert torch.isfinite(empty_output).all()

    # Subnet/phase context must remain visible even when there are no host
    # tokens to carry it into the Transformer.
    changed_context = model_input.context.clone()
    changed_context[:, 0, 0] = 0.0
    changed_context[:, 0, 1] = 1.0
    changed_output = encoder(empty_features, empty_indices, changed_context)
    assert not torch.allclose(empty_output, changed_output)

    host_count = model_input.host_features.shape[1]
    padding = torch.zeros((1, 2, 4), dtype=torch.float32)
    padding_indices = torch.zeros((1, 2), dtype=torch.long)
    padded_output = encoder(
        torch.cat((model_input.host_features, padding), dim=1),
        torch.cat((model_input.host_subnet_indices, padding_indices), dim=1),
        model_input.context,
        host_padding_mask=torch.cat(
            (torch.zeros((1, host_count), dtype=torch.bool), torch.ones((1, 2), dtype=torch.bool)),
            dim=1,
        ),
    )
    plain_output = encoder(
        model_input.host_features,
        model_input.host_subnet_indices,
        model_input.context,
    )
    assert torch.allclose(plain_output, padded_output, rtol=1e-5, atol=1e-6)


def test_variable_transformer_is_invariant_to_host_token_order(variable_env):
    observations, infos = variable_env.reset(seed=21)
    model_input = _variable_input(variable_env, observations, infos, "blue_agent_4")
    encoder = VariableCC4TransformerEncoder().eval()
    permutation = torch.arange(model_input.host_features.shape[1] - 1, -1, -1)

    original = encoder(
        model_input.host_features,
        model_input.host_subnet_indices,
        model_input.context,
    )
    reordered = encoder(
        model_input.host_features[:, permutation],
        model_input.host_subnet_indices[:, permutation],
        model_input.context,
    )
    assert torch.allclose(original, reordered, rtol=1e-5, atol=1e-6)


def test_variable_transformer_policy_mapping_and_checkpoint_round_trip(
    variable_env, tmp_path
):
    observations, infos = variable_env.reset(seed=21)
    for agent in variable_env.possible_agents:
        model_input = _variable_input(variable_env, observations, infos, agent)
        spec = AgentActionSpec.from_wrapper(variable_env, agent)
        policy = VariableIndependentPolicy(VariableCC4TransformerEncoder(), spec).eval()
        original_logits = policy(model_input)
        action_index = spec.select_greedy(original_logits[0], infos[agent]["action_mask"])

        assert original_logits.shape == (1, variable_env.action_space(agent).n)
        assert spec.label(action_index) == variable_env.action_labels(agent)[action_index]
        assert infos[agent]["action_mask"][action_index]

        checkpoint = tmp_path / f"{agent}_variable_transformer.pt"
        torch.save(policy.state_dict(), checkpoint)
        reloaded = VariableIndependentPolicy(
            VariableCC4TransformerEncoder(), spec
        ).eval()
        reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        assert torch.equal(original_logits, reloaded(model_input))


def test_three_variable_encoders_use_the_same_input_and_finite_policy_outputs(variable_env):
    observations, infos = variable_env.reset(seed=21)

    for agent in variable_env.possible_agents:
        adapter = CC4VariableHostAdapter(variable_env, agent)
        model_input = _variable_input(variable_env, observations, infos, agent)
        max_host_count = len(adapter.subnets) * 16
        encoders = (
            VariableCC4ZeroPaddingEncoder(max_host_count, len(adapter.subnets)),
            VariableCC4DeepSetsEncoder(),
            VariableCC4TransformerEncoder(),
        )
        spec = AgentActionSpec.from_wrapper(variable_env, agent)

        for encoder in encoders:
            policy = VariableIndependentPolicy(encoder, spec).eval()
            logits = policy(model_input)
            assert logits.shape == (1, variable_env.action_space(agent).n)
            assert torch.isfinite(logits).all()
            action_index = spec.select_greedy(logits[0], infos[agent]["action_mask"])
            assert infos[agent]["action_mask"][action_index]
            assert spec.label(action_index) == variable_env.action_labels(agent)[action_index]


def test_variable_padding_rejects_overflow_and_deep_sets_handles_empty_input(variable_env):
    observations, infos = variable_env.reset(seed=21)
    agent = "blue_agent_4"
    model_input = _variable_input(variable_env, observations, infos, agent)
    max_host_count = 3 * 16
    padding = VariableCC4ZeroPaddingEncoder(max_host_count, subnet_count=3)
    overflow_features = torch.zeros((1, max_host_count + 1, 4))
    overflow_indices = torch.zeros((1, max_host_count + 1), dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds padding maximum"):
        padding(overflow_features, overflow_indices, model_input.context)

    empty = VariableCC4DeepSetsEncoder().eval()
    empty_output = empty(
        torch.empty((1, 0, 4)),
        torch.empty((1, 0), dtype=torch.long),
        model_input.context,
    )
    assert empty_output.shape == (1, 128)
    assert torch.isfinite(empty_output).all()


@pytest.mark.parametrize("representation", ("zero_padding", "deep_sets", "transformer"))
def test_each_variable_representation_completes_masked_shared_reward_rollout(
    variable_env, representation
):
    variable_env.reset(seed=21)
    policies, adapters, builder = build_variable_independent_policies(
        variable_env, representation
    )
    result = CC4VariableRolloutRunner(
        variable_env, policies, adapters, builder
    ).run(max_environment_steps=4, seed=21)

    assert len(result.steps) == 4
    assert set(result.decisions_by_agent) == set(variable_env.possible_agents)
    assert all(result.decisions_by_agent[agent] for agent in variable_env.possible_agents)
    assert np.isclose(result.team_return, sum(step.team_reward for step in result.steps))
    for step in result.steps:
        assert np.allclose(list(step.per_agent_rewards.values()), step.team_reward)
        for agent, action_index in step.action_indices.items():
            assert action_index < variable_env.action_space(agent).n
            assert step.action_labels[agent] == variable_env.action_labels(agent)[action_index]
    for decisions in result.decisions_by_agent.values():
        for decision in decisions:
            assert bool(decision.action_mask[decision.action_index])
