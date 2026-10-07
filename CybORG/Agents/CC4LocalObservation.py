"""Parse CC4's official BlueFlatWrapper vector for an independent Blue policy.

The parser deliberately keeps the environment's communication-policy features,
but drops the trailing inter-agent message payload. Host slots follow the
wrapper's fixed sorted ordering; they are not interpreted as a host-presence
signal because the flat observation does not provide one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import SUBNET


NUM_SUBNETS = 9
MAX_HOSTS_PER_SUBNET = 16
NUM_MESSAGES = 4
MESSAGE_LENGTH = 8
MESSAGE_PAYLOAD_SIZE = NUM_MESSAGES * MESSAGE_LENGTH
SUBNET_BLOCK_SIZE = 3 * NUM_SUBNETS + 2 * MAX_HOSTS_PER_SUBNET

SUBNET_ORDER = tuple(sorted(subnet.value for subnet in SUBNET))


@dataclass(frozen=True)
class LocalObservation:
    """Structured, local input to one independent CC4 Blue policy.

    ``subnet_context`` has shape ``(n_local_subnets, 27)`` and contains the
    subnet one-hot, blocked-subnet vector, and communications-policy vector.
    ``host_events`` has shape ``(n_local_subnets, 16, 2)`` with malicious
    process and network-connection event flags. Empty slots remain zero and
    are not claimed to mean that a host is absent. ``action_mask`` is kept
    separate so the policy can mask currently unavailable actions.
    """

    mission_phase: int
    subnet_context: np.ndarray
    host_events: np.ndarray
    action_mask: np.ndarray


class CC4LocalObservationAdapter:
    """Decode one agent's observation from a ``BlueFlatWrapper`` instance."""

    def __init__(self, wrapper, agent_name: str):
        if agent_name not in wrapper.possible_agents:
            raise ValueError(f"Unknown Blue agent: {agent_name!r}")
        self.agent_name = agent_name
        self.subnets = tuple(wrapper.subnets(agent_name))
        self.host_names_by_subnet = tuple(
            tuple(
                hostname
                for hostname in wrapper.hosts(agent_name)
                if subnet in hostname and "router" not in hostname
            )
            for subnet in self.subnets
        )
        for subnet, host_names in zip(self.subnets, self.host_names_by_subnet):
            if subnet not in SUBNET_ORDER:
                raise ValueError(f"Unexpected CC4 subnet in wrapper metadata: {subnet!r}")
            if len(host_names) != MAX_HOSTS_PER_SUBNET:
                raise ValueError(
                    f"Expected {MAX_HOSTS_PER_SUBNET} fixed host slots for {subnet}, "
                    f"got {len(host_names)}"
                )

        self.unpadded_size = 1 + len(self.subnets) * SUBNET_BLOCK_SIZE + MESSAGE_PAYLOAD_SIZE

    def parse(
        self,
        observation: Sequence[int] | np.ndarray,
        action_mask: Sequence[bool] | np.ndarray,
    ) -> LocalObservation:
        """Return the local features, omitting messages and wrapper padding."""

        vector = np.asarray(observation)
        if vector.ndim != 1 or vector.size < self.unpadded_size:
            raise ValueError(
                f"{self.agent_name} observation must be a 1-D vector with at least "
                f"{self.unpadded_size} values; got shape {vector.shape}"
            )
        if not np.issubdtype(vector.dtype, np.integer):
            raise TypeError(f"Expected integer CC4 observation, got {vector.dtype}")

        # BlueFlatWrapper appends zero padding after its fixed message payload.
        expected_total = self.unpadded_size
        padding = vector[expected_total:]
        if padding.size and np.any(padding != 0):
            raise ValueError("Unexpected non-zero wrapper padding")

        phase = int(vector[0])
        if phase not in (0, 1, 2):
            raise ValueError(f"Unexpected CC4 mission phase: {phase}")

        n_local = len(self.subnets)
        context = np.zeros((n_local, 3 * NUM_SUBNETS), dtype=np.bool_)
        host_events = np.zeros(
            (n_local, MAX_HOSTS_PER_SUBNET, 2), dtype=np.bool_
        )
        cursor = 1
        for local_index, subnet in enumerate(self.subnets):
            subnet_one_hot = vector[cursor : cursor + NUM_SUBNETS]
            cursor += NUM_SUBNETS
            blocked = vector[cursor : cursor + NUM_SUBNETS]
            cursor += NUM_SUBNETS
            comms_policy = vector[cursor : cursor + NUM_SUBNETS]
            cursor += NUM_SUBNETS
            processes = vector[cursor : cursor + MAX_HOSTS_PER_SUBNET]
            cursor += MAX_HOSTS_PER_SUBNET
            connections = vector[cursor : cursor + MAX_HOSTS_PER_SUBNET]
            cursor += MAX_HOSTS_PER_SUBNET

            expected_subnet = np.zeros(NUM_SUBNETS, dtype=np.bool_)
            expected_subnet[SUBNET_ORDER.index(subnet)] = True
            if not np.array_equal(subnet_one_hot.astype(bool), expected_subnet):
                raise ValueError(
                    f"Subnet block order/content mismatch for {self.agent_name}: {subnet}"
                )

            context[local_index] = np.concatenate(
                (subnet_one_hot, blocked, comms_policy)
            ).astype(bool)
            host_events[local_index, :, 0] = processes.astype(bool)
            host_events[local_index, :, 1] = connections.astype(bool)

        mask = np.asarray(action_mask, dtype=np.bool_)
        if mask.ndim != 1 or mask.size == 0:
            raise ValueError(f"Invalid action mask shape: {mask.shape}")

        return LocalObservation(
            mission_phase=phase,
            subnet_context=context,
            host_events=host_events,
            action_mask=mask.copy(),
        )
