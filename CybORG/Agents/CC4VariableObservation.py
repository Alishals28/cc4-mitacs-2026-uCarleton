"""Variable actionable-host observations for the audited CC4 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from CybORG.Agents.CC4LocalObservation import (
    CC4LocalObservationAdapter,
    LocalObservation,
)


VARIABLE_HOST_FEATURES = 4


@dataclass(frozen=True)
class VariableHostObservation:
    """Policy-visible variable-host input derived from the wrapper interface.

    ``host_features`` contains one row per actionable host target, not one row
    per confirmed occupied host. The actionable set is derived from valid
    host-specific action labels and masks. Simulator host/session state is not
    consulted by this adapter.
    """

    mission_phase: int
    subnet_context: np.ndarray
    host_features: np.ndarray
    host_subnet_indices: np.ndarray
    actionable_hosts: tuple[str, ...]
    action_mask: np.ndarray
    action_labels: tuple[str, ...]


class CC4VariableHostAdapter:
    """Build one policy token per wrapper-audited actionable host target."""

    def __init__(self, wrapper, agent_name: str):
        self.wrapper = wrapper
        self.agent_name = agent_name
        self.fixed_adapter = CC4LocalObservationAdapter(wrapper, agent_name)
        self.subnets = self.fixed_adapter.subnets
        self.host_names_by_subnet = self.fixed_adapter.host_names_by_subnet
        self.host_slots = {
            hostname: (subnet_index, slot_index)
            for subnet_index, host_names in enumerate(self.host_names_by_subnet)
            for slot_index, hostname in enumerate(host_names)
        }
        self.host_names = tuple(self.host_slots)

    def _actionable_hosts(
        self, action_mask: Sequence[bool] | np.ndarray
    ) -> tuple[str, ...]:
        mask = np.asarray(action_mask, dtype=np.bool_)
        labels = tuple(self.wrapper.action_labels(self.agent_name))
        if mask.ndim != 1 or mask.size != len(labels):
            raise ValueError(
                f"Action mask shape {mask.shape} does not match {self.agent_name}'s labels"
            )

        actionable = {
            hostname
            for label, valid in zip(labels, mask)
            if valid
            for hostname in self.host_names
            if label.endswith(f" {hostname}")
        }
        return tuple(
            hostname
            for subnet_hosts in self.host_names_by_subnet
            for hostname in subnet_hosts
            if hostname in actionable
        )

    @staticmethod
    def _role_features(hostname: str) -> tuple[float, float]:
        if "_user_host_" in hostname:
            return 1.0, 0.0
        if "_server_host_" in hostname:
            return 0.0, 1.0
        raise ValueError(f"Unsupported CC4 candidate host slot: {hostname}")

    def parse(
        self,
        observation: Sequence[int] | np.ndarray,
        action_mask: Sequence[bool] | np.ndarray,
    ) -> VariableHostObservation:
        """Return actionable-host tokens using only wrapper-visible inputs."""

        fixed: LocalObservation = self.fixed_adapter.parse(observation, action_mask)
        actionable_hosts = self._actionable_hosts(action_mask)
        host_features = np.zeros(
            (len(actionable_hosts), VARIABLE_HOST_FEATURES), dtype=np.float32
        )
        host_subnet_indices = np.zeros(len(actionable_hosts), dtype=np.int64)

        for index, hostname in enumerate(actionable_hosts):
            subnet_index, slot_index = self.host_slots[hostname]
            host_features[index, :2] = fixed.host_events[subnet_index, slot_index]
            host_features[index, 2:] = self._role_features(hostname)
            host_subnet_indices[index] = subnet_index

        return VariableHostObservation(
            mission_phase=fixed.mission_phase,
            subnet_context=fixed.subnet_context.copy(),
            host_features=host_features,
            host_subnet_indices=host_subnet_indices,
            actionable_hosts=actionable_hosts,
            action_mask=fixed.action_mask.copy(),
            action_labels=tuple(self.wrapper.action_labels(self.agent_name)),
        )
