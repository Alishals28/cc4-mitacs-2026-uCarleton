import numpy as np
import pytest
import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4LocalObservation import CC4LocalObservationAdapter
from CybORG.Agents.CC4Representations import (
    AgentActionSpec,
    CC4FeatureSchema,
    CC4ModelInputBuilder,
    CC4TransformerEncoder,
    DeepSetsEncoder,
    IndependentPolicy,
    ZeroPaddingEncoder,
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


def test_common_model_input_preserves_agent_local_shapes(blue_flat_env):
    observations, infos = blue_flat_env.reset()

    for agent in blue_flat_env.agents:
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        model_input = CC4ModelInputBuilder(adapter).build(parsed)

        expected_subnets = 1 if agent != "blue_agent_4" else 3
        assert model_input.candidate_features.shape == (expected_subnets, 16, 4)
        assert model_input.context.shape == (expected_subnets, 30)
        assert np.array_equal(
            model_input.candidate_features[:, :, :2].numpy(),
            parsed.host_events.astype(np.float32),
        )


def test_common_feature_schema_names_only_observable_inputs():
    schema = CC4FeatureSchema()

    assert schema.candidate_width == 4
    assert schema.context_width == 30
    assert schema.candidate_features == (
        "process_alert",
        "connection_alert",
        "user_role",
        "server_role",
    )
    assert "received_message_bits" in schema.excluded_information
    assert "host_occupancy" in schema.excluded_information


def test_baselines_produce_fixed_representations_for_each_agent(blue_flat_env):
    observations, infos = blue_flat_env.reset()

    for agent in blue_flat_env.agents:
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        model_input = CC4ModelInputBuilder(adapter).build(parsed)
        candidate_features = model_input.candidate_features.unsqueeze(0)
        context = model_input.context.unsqueeze(0)

        zero_padding = ZeroPaddingEncoder(len(adapter.subnets))
        deep_sets = DeepSetsEncoder()
        assert zero_padding(candidate_features, context).shape == (1, 128)
        assert deep_sets(candidate_features, context).shape == (1, 128)


def test_deep_sets_is_invariant_to_candidate_slot_order(blue_flat_env):
    observations, infos = blue_flat_env.reset()
    agent = "blue_agent_4"
    adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
    parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
    model_input = CC4ModelInputBuilder(adapter).build(parsed)
    candidate_features = model_input.candidate_features.unsqueeze(0)
    context = model_input.context.unsqueeze(0)
    permutation = torch.tensor(list(reversed(range(16))))

    encoder = DeepSetsEncoder().eval()
    original = encoder(candidate_features, context)
    reordered = encoder(candidate_features[:, :, permutation, :], context)
    assert torch.allclose(original, reordered)


def test_transformer_produces_finite_cls_representation_for_all_zero_slots(blue_flat_env):
    observations, infos = blue_flat_env.reset()

    for agent in blue_flat_env.agents:
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        model_input = CC4ModelInputBuilder(adapter).build(parsed)
        encoder = CC4TransformerEncoder().eval()
        candidate_features = torch.zeros_like(model_input.candidate_features).unsqueeze(0)
        context = model_input.context.unsqueeze(0)

        representation = encoder(candidate_features, context)

        assert representation.shape == (1, 128)
        assert torch.isfinite(representation).all()


def test_transformer_reuses_weights_across_one_and_three_local_subnets(blue_flat_env):
    observations, infos = blue_flat_env.reset()
    encoder = CC4TransformerEncoder().eval()
    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in encoder.state_dict().items()
    }

    for agent in ("blue_agent_0", "blue_agent_4"):
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        model_input = CC4ModelInputBuilder(adapter).build(parsed)
        representation = encoder(
            model_input.candidate_features.unsqueeze(0),
            model_input.context.unsqueeze(0),
        )

        expected_subnets = 1 if agent == "blue_agent_0" else 3
        assert model_input.candidate_features.shape == (expected_subnets, 16, 4)
        assert representation.shape == (1, 128)
        assert torch.isfinite(representation).all()

    for name, parameter in encoder.state_dict().items():
        assert torch.equal(parameter, initial_state[name])


def test_transformer_policy_uses_agent_specific_action_mapping(blue_flat_env):
    observations, infos = blue_flat_env.reset()

    for agent in blue_flat_env.agents:
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        model_input = CC4ModelInputBuilder(adapter).build(parsed)
        spec = AgentActionSpec.from_wrapper(blue_flat_env, agent)
        policy = IndependentPolicy(CC4TransformerEncoder(), spec)
        logits = policy(
            type(model_input)(
                model_input.candidate_features.unsqueeze(0),
                model_input.context.unsqueeze(0),
            )
        )

        action_index = spec.select_greedy(logits[0], infos[agent]["action_mask"])
        assert logits.shape == (1, blue_flat_env.action_space(agent).n)
        assert spec.label(action_index) == blue_flat_env.action_labels(agent)[action_index]


def test_encoders_and_policies_round_trip_through_checkpoints(blue_flat_env, tmp_path):
    observations, infos = blue_flat_env.reset()

    for agent in blue_flat_env.agents:
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        model_input = CC4ModelInputBuilder(adapter).build(parsed)
        batched_input = type(model_input)(
            model_input.candidate_features.unsqueeze(0),
            model_input.context.unsqueeze(0),
        )
        action_spec = AgentActionSpec.from_wrapper(blue_flat_env, agent)

        encoder_factories = (
            lambda: ZeroPaddingEncoder(len(adapter.subnets)),
            DeepSetsEncoder,
            CC4TransformerEncoder,
        )
        for index, encoder_factory in enumerate(encoder_factories):
            encoder = encoder_factory().eval()
            original_representation = encoder(
                batched_input.candidate_features, batched_input.context
            )
            checkpoint_path = tmp_path / f"{agent}_encoder_{index}.pt"
            torch.save(encoder.state_dict(), checkpoint_path)

            reloaded_encoder = encoder_factory().eval()
            reloaded_encoder.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            reloaded_representation = reloaded_encoder(
                batched_input.candidate_features, batched_input.context
            )
            assert torch.equal(original_representation, reloaded_representation)

            policy = IndependentPolicy(encoder_factory(), action_spec).eval()
            original_logits = policy(batched_input)
            policy_path = tmp_path / f"{agent}_policy_{index}.pt"
            torch.save(policy.state_dict(), policy_path)

            reloaded_policy = IndependentPolicy(encoder_factory(), action_spec).eval()
            reloaded_policy.load_state_dict(torch.load(policy_path, weights_only=True))
            reloaded_logits = reloaded_policy(batched_input)
            assert torch.equal(original_logits, reloaded_logits)


def test_action_spec_uses_each_agents_own_mapping(blue_flat_env):
    observations, infos = blue_flat_env.reset()

    for agent in blue_flat_env.agents:
        adapter = CC4LocalObservationAdapter(blue_flat_env, agent)
        parsed = adapter.parse(observations[agent], infos[agent]["action_mask"])
        model_input = CC4ModelInputBuilder(adapter).build(parsed)
        spec = AgentActionSpec.from_wrapper(blue_flat_env, agent)
        policy = IndependentPolicy(ZeroPaddingEncoder(len(adapter.subnets)), spec)
        logits = policy(
            type(model_input)(
                model_input.candidate_features.unsqueeze(0),
                model_input.context.unsqueeze(0),
            )
        )

        assert logits.shape == (1, blue_flat_env.action_space(agent).n)
        action_index = spec.select_greedy(logits[0], infos[agent]["action_mask"])
        assert infos[agent]["action_mask"][action_index]
        assert spec.label(action_index) == blue_flat_env.action_labels(agent)[action_index]
