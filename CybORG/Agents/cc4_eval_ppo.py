"""Deterministically evaluate saved CC4 variable-host PPO checkpoints.

Example from the repository root::

    python -m CybORG.Agents.cc4_eval_ppo \
        --checkpoint-dir runs/cc4_ppo_pilot/run_20261007_183750 \
        --representations all --episodes 10 --episode-steps 500 \
        --seed 1000 --device cuda

The evaluation seeds should be held out from training. Policies use masked
greedy action selection, and each shared Blue reward is counted once per tick.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4Rollout import (
    CC4VariableRolloutRunner,
    build_variable_independent_policies,
)
from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


REPRESENTATIONS = ("zero_padding", "deep_sets", "transformer")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--representations", default="all")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.episode_steps < 2:
        parser.error("episodes must be positive and episode-steps must be at least 2")
    if args.representations == "all":
        args.representations = REPRESENTATIONS
    else:
        args.representations = tuple(
            item.strip() for item in args.representations.split(",") if item.strip()
        )
        unknown = set(args.representations) - set(REPRESENTATIONS)
        if unknown or not args.representations:
            parser.error(f"unknown or empty representation list: {sorted(unknown)}")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this Python environment")
    return args


def _make_wrapper(seed: int, episode_steps: int) -> BlueFlatWrapper:
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
        steps=episode_steps,
    )
    wrapper = BlueFlatWrapper(
        CybORG(scenario_generator=scenario, seed=seed), pad_spaces=True
    )
    wrapper.reset(seed=seed)
    return wrapper


def _load_policy_checkpoints(checkpoint_dir, representation, policies, device):
    for agent, policy in policies.items():
        path = checkpoint_dir / representation / f"{agent}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {representation}/{agent}: {path}")
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # Compatibility with older PyTorch releases.
            payload = torch.load(path, map_location=device)
        if payload.get("representation") != representation or payload.get("agent") != agent:
            raise ValueError(f"Checkpoint identity mismatch in {path}")
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
        policy.eval()


def evaluate(args) -> Path:
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    device = torch.device(args.device)
    output_dir = args.output_dir or (checkpoint_dir / "evaluation")
    run_dir = output_dir / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "metrics.csv"
    fields = ("representation", "episode", "seed", "team_return", "environment_steps",
              "blue_policy_decisions", "invalid_actions", "runtime_seconds")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for representation in args.representations:
            first_seed = args.seed
            wrapper = _make_wrapper(first_seed, args.episode_steps)
            policies, adapters, input_builder = build_variable_independent_policies(
                wrapper, representation, device=device
            )
            _load_policy_checkpoints(checkpoint_dir, representation, policies, device)
            runner = CC4VariableRolloutRunner(wrapper, policies, adapters, input_builder)
            for episode in range(args.episodes):
                seed = args.seed + episode
                started = time.perf_counter()
                result = runner.run(max_environment_steps=args.episode_steps, seed=seed)
                elapsed = time.perf_counter() - started
                # The rollout runner checks every selected index against its
                # current action mask and raises before stepping if invalid.
                invalid_actions = 0
                decisions = sum(len(items) for items in result.decisions_by_agent.values())
                row = {
                    "representation": representation,
                    "episode": episode + 1,
                    "seed": seed,
                    "team_return": result.team_return,
                    "environment_steps": len(result.steps),
                    "blue_policy_decisions": decisions,
                    "invalid_actions": invalid_actions,
                    "runtime_seconds": round(elapsed, 3),
                }
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"{representation} eval {episode + 1}/{args.episodes}: "
                    f"seed={seed}, return={result.team_return:.3f}, "
                    f"steps={len(result.steps)}, decisions={decisions}, "
                    f"invalid_actions={invalid_actions}, elapsed={elapsed:.1f}s"
                )
    print(f"Evaluation metrics written to: {csv_path.resolve()}")
    return run_dir


def main(argv=None):
    evaluate(_parse_args(argv))


if __name__ == "__main__":
    main()
