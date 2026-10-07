"""CC4 representation baselines and agent-specific action handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from CybORG.Agents.CC4LocalObservation import CC4LocalObservationAdapter, LocalObservation


NUM_PHASES = 3
CANDIDATE_FEATURES = 4
CONTEXT_FEATURES = NUM_PHASES + 3 * 9
MAX_CANDIDATE_SLOTS = 16


@dataclass(frozen=True)
class CC4FeatureSchema:
    """Named, observable feature layout shared by all CC4 representations.

    The numbered subnet positions describe tensor layout, not a standalone
    global ownership mapping. The adapter's subnet metadata and one-hot values
    determine which actual subnet a local row represents for each agent.
    """

    candidate_features: tuple[str, ...] = (
        "process_alert",
        "connection_alert",
        "user_role",
        "server_role",
    )
    context_features: tuple[str, ...] = (
        "mission_phase_0",
        "mission_phase_1",
        "mission_phase_2",
        *(f"subnet_identity_{index}" for index in range(9)),
        *(f"blocked_subnet_{index}" for index in range(9)),
        *(f"communication_policy_{index}" for index in range(9)),
    )
    excluded_information: tuple[str, ...] = (
        "received_message_bits",
        "host_occupancy",
        "raw_simulator_state",
        "red_or_green_internal_state",
    )

    @property
    def candidate_width(self) -> int:
        return len(self.candidate_features)

    @property
    def context_width(self) -> int:
        return len(self.context_features)


@dataclass(frozen=True)
class CC4ModelInput:
    """Observable tensors supplied to one independent Blue policy."""

    candidate_features: Tensor
    context: Tensor


@dataclass(frozen=True)
class VariableCC4ModelInput:
    """Observable variable-host tensors for one independent Blue policy."""

    host_features: Tensor
    host_subnet_indices: Tensor
    context: Tensor
    host_padding_mask: Tensor | None = None


class VariableCC4InputBuilder:
    """Convert one variable-host observation into the shared model input."""

    def build(
        self,
        observation,
        device: torch.device | str | None = None,
    ) -> VariableCC4ModelInput:
        host_features = torch.as_tensor(observation.host_features, dtype=torch.float32)
        host_subnet_indices = torch.as_tensor(
            observation.host_subnet_indices, dtype=torch.long
        )
        phase = torch.zeros(NUM_PHASES, dtype=torch.float32)
        phase[observation.mission_phase] = 1.0
        phase_context = torch.cat(
            (
                phase.expand(observation.subnet_context.shape[0], -1),
                torch.as_tensor(observation.subnet_context, dtype=torch.float32),
            ),
            dim=-1,
        )
        if device is not None:
            host_features = host_features.to(device)
            host_subnet_indices = host_subnet_indices.to(device)
            phase_context = phase_context.to(device)
        return VariableCC4ModelInput(
            host_features=host_features,
            host_subnet_indices=host_subnet_indices,
            context=phase_context,
        )


class CC4ModelInputBuilder:
    """Build common model features from one agent's local adapter output."""

    schema = CC4FeatureSchema()

    def __init__(self, adapter: CC4LocalObservationAdapter):
        self.adapter = adapter
        self._role_features = self._build_role_features()

    def _build_role_features(self) -> Tensor:
        roles = []
        for host_names in self.adapter.host_names_by_subnet:
            subnet_roles = []
            for hostname in host_names:
                if "_user_host_" in hostname:
                    subnet_roles.append((1.0, 0.0))
                elif "_server_host_" in hostname:
                    subnet_roles.append((0.0, 1.0))
                else:
                    raise ValueError(f"Unsupported CC4 candidate host slot: {hostname}")
            roles.append(subnet_roles)
        return torch.tensor(roles, dtype=torch.float32)

    def build(
        self,
        observation: LocalObservation,
        device: torch.device | str | None = None,
    ) -> CC4ModelInput:
        """Return candidate-slot and local-context features without host inference."""

        event_features = torch.as_tensor(
            observation.host_events.astype(np.float32), dtype=torch.float32
        )
        candidate_features = torch.cat(
            (event_features, self._role_features.to(event_features.device)), dim=-1
        )

        phase = torch.zeros(NUM_PHASES, dtype=torch.float32, device=event_features.device)
        phase[observation.mission_phase] = 1.0
        phase = phase.expand(len(self.adapter.subnets), -1)
        context = torch.cat(
            (phase, torch.as_tensor(observation.subnet_context, dtype=torch.float32)),
            dim=-1,
        )

        if device is not None:
            candidate_features = candidate_features.to(device)
            context = context.to(device)

        return CC4ModelInput(candidate_features, context)


class _CC4Encoder(nn.Module):
    output_size: int

    @staticmethod
    def _validate_inputs(candidate_features: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        if candidate_features.ndim == 3:
            candidate_features = candidate_features.unsqueeze(0)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if candidate_features.ndim != 4 or context.ndim != 3:
            raise ValueError("CC4 features must be [batch, subnets, slots, features]")
        if candidate_features.shape[:2] != context.shape[:2]:
            raise ValueError("Candidate features and context must have matching subnet axes")
        if candidate_features.shape[2] != MAX_CANDIDATE_SLOTS:
            raise ValueError("CC4 input must retain all 16 candidate host slots")
        if candidate_features.shape[3] != CANDIDATE_FEATURES:
            raise ValueError("Unexpected candidate feature width")
        if context.shape[2] != CONTEXT_FEATURES:
            raise ValueError("Unexpected CC4 context feature width")
        return candidate_features, context


class ZeroPaddingEncoder(_CC4Encoder):
    """Fixed-slot baseline using a flattened local candidate-slot representation."""

    def __init__(self, subnet_count: int, hidden_size: int = 128):
        super().__init__()
        self.subnet_count = subnet_count
        input_size = subnet_count * (
            MAX_CANDIDATE_SLOTS * CANDIDATE_FEATURES + CONTEXT_FEATURES
        )
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_size = hidden_size

    def forward(self, candidate_features: Tensor, context: Tensor) -> Tensor:
        candidate_features, context = self._validate_inputs(candidate_features, context)
        if candidate_features.shape[1] != self.subnet_count:
            raise ValueError("Input subnet count does not match this agent's encoder")
        flattened = torch.cat(
            (candidate_features.flatten(start_dim=2), context), dim=-1
        ).flatten(start_dim=1)
        return self.network(flattened)


class DeepSetsEncoder(_CC4Encoder):
    """Permutation-invariant baseline over all local candidate slots."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        token_size = CANDIDATE_FEATURES + CONTEXT_FEATURES
        self.phi = nn.Sequential(
            nn.Linear(token_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.output_size = hidden_size

    def forward(self, candidate_features: Tensor, context: Tensor) -> Tensor:
        candidate_features, context = self._validate_inputs(candidate_features, context)
        expanded_context = context.unsqueeze(2).expand(-1, -1, MAX_CANDIDATE_SLOTS, -1)
        tokens = torch.cat((candidate_features, expanded_context), dim=-1)
        tokens = tokens.flatten(start_dim=1, end_dim=2)
        encoded = self.phi(tokens)
        pooled = encoded.max(dim=1).values
        return self.readout(pooled)


class CC4TransformerEncoder(_CC4Encoder):
    """CC2-style candidate-slot Transformer with a fixed ``[CLS]`` output."""

    def __init__(
        self,
        hidden_size: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")

        self.candidate_projection = nn.Linear(CANDIDATE_FEATURES, hidden_size)
        self.context_projection = nn.Linear(CONTEXT_FEATURES, hidden_size)
        self.slot_embedding = nn.Parameter(torch.randn(MAX_CANDIDATE_SLOTS, hidden_size) * 0.02)
        self.subnet_embedding = nn.Parameter(torch.randn(3, hidden_size) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_size = hidden_size

    def forward(self, candidate_features: Tensor, context: Tensor) -> Tensor:
        candidate_features, context = self._validate_inputs(candidate_features, context)
        batch_size, subnet_count = candidate_features.shape[:2]
        if subnet_count > self.subnet_embedding.shape[0]:
            raise ValueError("CC4 Transformer received more local subnets than supported")

        tokens = self.candidate_projection(candidate_features)
        tokens = tokens + self.slot_embedding.view(1, 1, MAX_CANDIDATE_SLOTS, -1)
        tokens = tokens + self.subnet_embedding[:subnet_count].view(1, subnet_count, 1, -1)
        tokens = tokens + self.context_projection(context).unsqueeze(2)
        tokens = tokens.flatten(start_dim=1, end_dim=2)

        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = self.transformer(torch.cat((cls, tokens), dim=1))
        return self.output_norm(encoded[:, 0])


class VariableCC4TransformerEncoder(nn.Module):
    """Transformer over actionable-host tokens with no slot-position embeddings."""

    def __init__(
        self,
        hidden_size: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_local_subnets: int = 3,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        if max_local_subnets < 1:
            raise ValueError("max_local_subnets must be positive")

        self.host_projection = nn.Linear(CANDIDATE_FEATURES, hidden_size)
        self.context_projection = nn.Linear(CONTEXT_FEATURES, hidden_size)
        self.subnet_embedding = nn.Parameter(
            torch.randn(max_local_subnets, hidden_size) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_size = hidden_size

    @staticmethod
    def _validate_inputs(
        host_features: Tensor,
        host_subnet_indices: Tensor,
        context: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if host_features.ndim == 2:
            host_features = host_features.unsqueeze(0)
        if host_subnet_indices.ndim == 1:
            host_subnet_indices = host_subnet_indices.unsqueeze(0)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if host_features.ndim != 3:
            raise ValueError("Variable host features must be [batch, hosts, features]")
        if host_subnet_indices.ndim != 2 or context.ndim != 3:
            raise ValueError("Variable subnet indices/context have invalid dimensions")
        if host_features.shape[:2] != host_subnet_indices.shape:
            raise ValueError("Host features and subnet indices must have matching axes")
        if host_features.shape[-1] != CANDIDATE_FEATURES:
            raise ValueError("Unexpected variable host feature width")
        if context.shape[-1] != CONTEXT_FEATURES:
            raise ValueError("Unexpected CC4 context feature width")
        if host_subnet_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("Host subnet indices must be integer tensors")
        return host_features, host_subnet_indices.long(), context

    def forward(
        self,
        host_features: Tensor,
        host_subnet_indices: Tensor,
        context: Tensor,
        host_padding_mask: Tensor | None = None,
    ) -> Tensor:
        host_features, host_subnet_indices, context = self._validate_inputs(
            host_features, host_subnet_indices, context
        )
        batch_size, host_count = host_features.shape[:2]
        subnet_count = context.shape[1]
        if subnet_count > self.subnet_embedding.shape[0]:
            raise ValueError("Too many local subnets for variable Transformer")
        if host_subnet_indices.numel() and (
            int(host_subnet_indices.min()) < 0
            or int(host_subnet_indices.max()) >= subnet_count
        ):
            raise ValueError("Host subnet index is outside the local context")

        if host_padding_mask is None:
            host_padding_mask = torch.zeros(
                (batch_size, host_count), dtype=torch.bool, device=host_features.device
            )
        else:
            host_padding_mask = torch.as_tensor(
                host_padding_mask, dtype=torch.bool, device=host_features.device
            )
            if host_padding_mask.ndim == 1:
                host_padding_mask = host_padding_mask.unsqueeze(0)
            if host_padding_mask.shape != (batch_size, host_count):
                raise ValueError("Host padding mask must match [batch, hosts]")

        # Keep every observed local subnet's context in the sequence even if
        # it currently has no actionable host token. Otherwise the encoder
        # would silently discard phase/block/communication context for that
        # subnet (and all context when the host set is empty).
        subnet_indices = torch.arange(subnet_count, device=context.device)
        context_tokens = self.context_projection(context) + self.subnet_embedding[
            subnet_indices
        ].unsqueeze(0)

        if host_count:
            context_for_hosts = torch.gather(
                context,
                1,
                host_subnet_indices.unsqueeze(-1).expand(-1, -1, context.shape[-1]),
            )
            tokens = self.host_projection(host_features)
            tokens = tokens + self.context_projection(context_for_hosts)
            tokens = tokens + self.subnet_embedding[host_subnet_indices]
        else:
            tokens = host_features.new_empty((batch_size, 0, self.output_size))

        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((cls, context_tokens, tokens), dim=1)
        sequence_mask = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1 + subnet_count),
                    dtype=torch.bool,
                    device=host_features.device,
                ),
                host_padding_mask,
            ),
            dim=1,
        )
        encoded = self.transformer(sequence, src_key_padding_mask=sequence_mask)
        return self.output_norm(encoded[:, 0])


class VariableCC4ZeroPaddingEncoder(nn.Module):
    """Variable-host padding baseline with an explicit padding indicator."""

    def __init__(
        self,
        max_host_count: int,
        subnet_count: int,
        hidden_size: int = 128,
    ):
        super().__init__()
        if max_host_count < 1 or subnet_count < 1:
            raise ValueError("Variable padding dimensions must be positive")
        self.max_host_count = max_host_count
        self.subnet_count = subnet_count
        token_width = CANDIDATE_FEATURES + 1 + CONTEXT_FEATURES
        input_size = max_host_count * token_width + subnet_count * CONTEXT_FEATURES
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_size = hidden_size

    def forward(
        self,
        host_features: Tensor,
        host_subnet_indices: Tensor,
        context: Tensor,
        host_padding_mask: Tensor | None = None,
    ) -> Tensor:
        host_features, host_subnet_indices, context = VariableCC4TransformerEncoder._validate_inputs(
            host_features, host_subnet_indices, context
        )
        batch_size, host_count = host_features.shape[:2]
        if host_count > self.max_host_count:
            raise ValueError(
                f"Variable host count {host_count} exceeds padding maximum "
                f"{self.max_host_count}"
            )
        if context.shape[1] != self.subnet_count:
            raise ValueError("Input subnet count does not match padding encoder")
        if host_subnet_indices.numel() and int(host_subnet_indices.max()) >= self.subnet_count:
            raise ValueError("Host subnet index is outside the padding context")

        if host_padding_mask is None:
            valid_hosts = torch.ones(
                (batch_size, host_count), dtype=torch.bool, device=host_features.device
            )
        else:
            host_padding_mask = torch.as_tensor(
                host_padding_mask, dtype=torch.bool, device=host_features.device
            )
            if host_padding_mask.ndim == 1:
                host_padding_mask = host_padding_mask.unsqueeze(0)
            if host_padding_mask.shape != (batch_size, host_count):
                raise ValueError("Host padding mask must match [batch, hosts]")
            valid_hosts = ~host_padding_mask

        padded_features = host_features.new_zeros(
            (batch_size, self.max_host_count, CANDIDATE_FEATURES + 1)
        )
        padded_features[:, :, CANDIDATE_FEATURES] = 1.0
        if host_count:
            padded_features[:, :host_count, :CANDIDATE_FEATURES] = (
                host_features * valid_hosts.unsqueeze(-1)
            )
            padded_features[:, :host_count, CANDIDATE_FEATURES] = (
                ~valid_hosts
            ).to(host_features.dtype)
            host_context = torch.gather(
                context,
                1,
                host_subnet_indices.unsqueeze(-1).expand(-1, -1, context.shape[-1]),
            )
            host_context = host_context * valid_hosts.unsqueeze(-1)
        else:
            host_context = context.new_zeros((batch_size, 0, CONTEXT_FEATURES))
        padded_context = context.new_zeros(
            (batch_size, self.max_host_count, CONTEXT_FEATURES)
        )
        padded_context[:, :host_count] = host_context
        tokens = torch.cat((padded_features, padded_context), dim=-1)
        flat = torch.cat((tokens.flatten(start_dim=1), context.flatten(start_dim=1)), dim=-1)
        return self.network(flat)


class VariableCC4DeepSetsEncoder(nn.Module):
    """Permutation-invariant encoder over real actionable-host tokens."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        token_width = CANDIDATE_FEATURES + CONTEXT_FEATURES
        self.phi = nn.Sequential(
            nn.Linear(token_width, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(CONTEXT_FEATURES, hidden_size),
            nn.ReLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )
        self.empty_host_pool = nn.Parameter(torch.zeros(hidden_size))
        self.output_size = hidden_size

    def forward(
        self,
        host_features: Tensor,
        host_subnet_indices: Tensor,
        context: Tensor,
        host_padding_mask: Tensor | None = None,
    ) -> Tensor:
        host_features, host_subnet_indices, context = VariableCC4TransformerEncoder._validate_inputs(
            host_features, host_subnet_indices, context
        )
        batch_size, host_count = host_features.shape[:2]
        if host_subnet_indices.numel() and int(host_subnet_indices.max()) >= context.shape[1]:
            raise ValueError("Host subnet index is outside the Deep Sets context")
        if host_padding_mask is None:
            valid_hosts = torch.ones(
                (batch_size, host_count), dtype=torch.bool, device=host_features.device
            )
        else:
            host_padding_mask = torch.as_tensor(
                host_padding_mask, dtype=torch.bool, device=host_features.device
            )
            if host_padding_mask.ndim == 1:
                host_padding_mask = host_padding_mask.unsqueeze(0)
            if host_padding_mask.shape != (batch_size, host_count):
                raise ValueError("Host padding mask must match [batch, hosts]")
            valid_hosts = ~host_padding_mask
        context_summary = context.mean(dim=1)
        global_context = self.context_projection(context_summary)
        if host_count:
            host_context = torch.gather(
                context,
                1,
                host_subnet_indices.unsqueeze(-1).expand(-1, -1, context.shape[-1]),
            )
            token_features = torch.cat((host_features, host_context), dim=-1)
            encoded_hosts = self.phi(token_features)
            pooled = (encoded_hosts * valid_hosts.unsqueeze(-1)).sum(dim=1)
            pooled = torch.where(
                valid_hosts.any(dim=1, keepdim=True),
                pooled,
                self.empty_host_pool.expand(batch_size, -1),
            )
        else:
            pooled = self.empty_host_pool.expand(batch_size, -1)
        return self.readout(torch.cat((pooled, global_context), dim=-1))


@dataclass(frozen=True)
class AgentActionSpec:
    """Agent-local action labels used alongside a padded action mask."""

    agent_name: str
    labels: tuple[str, ...]

    @classmethod
    def from_wrapper(cls, wrapper, agent_name: str) -> "AgentActionSpec":
        labels = tuple(wrapper.action_labels(agent_name))
        if len(labels) != wrapper.action_space(agent_name).n:
            raise ValueError(f"Action labels do not match the space for {agent_name}")
        return cls(agent_name=agent_name, labels=labels)

    @property
    def action_size(self) -> int:
        return len(self.labels)

    def label(self, action_index: int) -> str:
        if not 0 <= action_index < self.action_size:
            raise IndexError(f"Action index {action_index} is invalid for {self.agent_name}")
        return self.labels[action_index]

    def mask_logits(self, logits: Tensor, action_mask: Sequence[bool] | Tensor) -> Tensor:
        if logits.ndim < 1 or logits.shape[-1] != self.action_size:
            raise ValueError(f"Action logits and mask do not match {self.agent_name}'s action mapping")
        if not logits.is_floating_point():
            raise TypeError("Action logits must be floating point")

        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
        if mask.ndim < 1 or mask.shape[-1] != self.action_size:
            raise ValueError(f"Action logits and mask do not match {self.agent_name}'s action mapping")
        try:
            mask = torch.broadcast_to(mask, logits.shape)
        except RuntimeError as error:
            raise ValueError(
                f"Action mask shape {tuple(mask.shape)} cannot broadcast to logits "
                f"shape {tuple(logits.shape)} for {self.agent_name}"
            ) from error
        if not torch.all(mask.any(dim=-1)):
            raise ValueError(
                f"All actions are masked for at least one {self.agent_name} batch item"
            )
        if not torch.isfinite(logits.masked_select(mask)).all():
            raise ValueError(f"Valid action logits must be finite for {self.agent_name}")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def masked_distribution(
        self, logits: Tensor, action_mask: Sequence[bool] | Tensor
    ) -> torch.distributions.Categorical:
        """Categorical policy whose probability is zero for every invalid action."""

        return torch.distributions.Categorical(
            logits=self.mask_logits(logits, action_mask)
        )

    def sample_action(
        self, logits: Tensor, action_mask: Sequence[bool] | Tensor
    ) -> tuple[int, Tensor]:
        """Sample one action and return its index and log probability."""

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("sample_action expects logits with shape [actions] or [1, actions]")
        distribution = self.masked_distribution(logits, action_mask)
        action = distribution.sample()
        return int(action.item()), distribution.log_prob(action).squeeze(0)

    def select_greedy(self, logits: Tensor, action_mask: Sequence[bool] | Tensor) -> int:
        if logits.ndim == 2 and logits.shape[0] != 1:
            raise ValueError("select_greedy accepts one policy observation at a time")
        if logits.ndim not in (1, 2):
            raise ValueError("select_greedy expects logits with shape [actions] or [1, actions]")
        masked = self.mask_logits(logits, action_mask)
        return int(torch.argmax(masked, dim=-1).reshape(-1)[0].item())

    def sample_uniform_valid_action(
        self,
        action_mask: Sequence[bool] | Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> int:
        """Sample uniformly from valid actions, for DQN epsilon exploration."""

        mask = torch.as_tensor(action_mask, dtype=torch.bool, device="cpu")
        if mask.ndim != 1 or mask.numel() != self.action_size:
            raise ValueError(f"Action mask shape does not match {self.agent_name}'s action space")
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError(f"All actions are masked for {self.agent_name}")
        rng_device = generator.device if generator is not None else torch.device("cpu")
        selected = torch.randint(
            valid_indices.numel(), (), generator=generator, device=rng_device
        ).item()
        return int(valid_indices[selected].item())

    def select_epsilon_greedy(
        self,
        q_values: Tensor,
        action_mask: Sequence[bool] | Tensor,
        epsilon: float,
        *,
        generator: torch.Generator | None = None,
    ) -> int:
        """Choose a valid random action with probability epsilon, else masked argmax."""

        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        rng_device = generator.device if generator is not None else torch.device("cpu")
        if torch.rand((), generator=generator, device=rng_device).item() < float(epsilon):
            return self.sample_uniform_valid_action(action_mask, generator=generator)
        return self.select_greedy(q_values, action_mask)


class IndependentPolicy(nn.Module):
    """One independently parameterized policy head for one Blue agent."""

    def __init__(self, encoder: _CC4Encoder, action_spec: AgentActionSpec):
        super().__init__()
        self.encoder = encoder
        self.action_spec = action_spec
        self.policy_head = nn.Linear(encoder.output_size, action_spec.action_size)

    def forward(self, model_input: CC4ModelInput) -> Tensor:
        representation = self.encoder(
            model_input.candidate_features, model_input.context
        )
        return self.policy_head(representation)


class VariableIndependentPolicy(nn.Module):
    """Independent actor-critic policy for a variable-host representation."""

    def __init__(
        self,
        encoder: nn.Module,
        action_spec: AgentActionSpec,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_spec = action_spec
        self.policy_head = nn.Linear(encoder.output_size, action_spec.action_size)
        self.value_head = nn.Linear(encoder.output_size, 1)

    def forward(self, model_input: VariableCC4ModelInput) -> Tensor:
        representation = self.encoder(
            model_input.host_features,
            model_input.host_subnet_indices,
            model_input.context,
            host_padding_mask=model_input.host_padding_mask,
        )
        return self.policy_head(representation)

    def evaluate(self, model_input: VariableCC4ModelInput) -> tuple[Tensor, Tensor]:
        """Return policy logits and one critic value per input observation."""

        representation = self.encoder(
            model_input.host_features,
            model_input.host_subnet_indices,
            model_input.context,
            host_padding_mask=model_input.host_padding_mask,
        )
        return self.policy_head(representation), self.value_head(representation).squeeze(-1)
