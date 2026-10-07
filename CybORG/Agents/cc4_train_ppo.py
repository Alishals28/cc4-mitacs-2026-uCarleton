"""Train matched PPO pilots for CC4's three variable-host representations.

Example from the repository root::

    python -m CybORG.Agents.cc4_train_ppo --episodes 3 --device cuda

By default this trains all three representations on the same environment seed
sequence. Results are pilot metrics; use more seeds and the official evaluation
protocol before drawing comparative conclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import time

import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.CC4Rollout import build_variable_independent_policies
from CybORG.Agents.CC4Training import CC4VariablePPOTrainer, PPOConfig
from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


REPRESENTATIONS = ("zero_padding", "deep_sets", "transformer")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", default="all")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--torch-seed", type=int, default=123)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cc4_variable_ppo"))
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


def run_pilot(args) -> Path:
    device = torch.device(args.device)
    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
    )
    manifest = {
        "representations": list(args.representations),
        "episodes_per_representation": args.episodes,
        "episode_steps": args.episode_steps,
        "environment_seed_sequence": [args.seed + i for i in range(args.episodes)],
        "torch_seed": args.torch_seed,
        "device": str(device),
        "ppo_config": config.__dict__,
        "reward": "official shared Blue reward; each independent agent trains on its copy",
        "token_definition": "wrapper-actionable host targets from the action mask",
    }
    (run_dir / "config.json").write_text(json.dumps(manifest, indent=2))
    csv_path = run_dir / "metrics.csv"
    fields = (
        "representation",
        "episode",
        "seed",
        "team_return",
        "environment_steps",
        "policy_decisions",
        "mean_loss",
        "optimizer_updates",
        "runtime_seconds",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for representation in args.representations:
            torch.manual_seed(args.torch_seed)
            wrapper = _make_wrapper(args.seed, args.episode_steps)
            policies, adapters, input_builder = build_variable_independent_policies(
                wrapper, representation, device=device
            )
            trainer = CC4VariablePPOTrainer(
                wrapper, policies, adapters, input_builder, config
            )
            started = time.perf_counter()
            for episode in range(args.episodes):
                episode_seed = args.seed + episode
                torch.manual_seed(args.torch_seed + episode)
                metrics = trainer.train_episode(seed=episode_seed)
                losses = [
                    metrics[f"{agent}/loss"] for agent in policies
                ]
                updates = sum(
                    int(metrics[f"{agent}/optimizer_updates"]) for agent in policies
                )
                elapsed = time.perf_counter() - started
                writer.writerow(
                    {
                        "representation": representation,
                        "episode": episode + 1,
                        "seed": episode_seed,
                        "team_return": metrics["episode/team_return"],
                        "environment_steps": int(metrics["episode/environment_steps"]),
                        "policy_decisions": int(metrics["episode/decisions"]),
                        "mean_loss": sum(losses) / len(losses),
                        "optimizer_updates": updates,
                        "runtime_seconds": round(elapsed, 3),
                    }
                )
                csv_file.flush()
                print(
                    f"{representation} episode {episode + 1}/{args.episodes}: "
                    f"return={metrics['episode/team_return']:.3f}, "
                    f"steps={int(metrics['episode/environment_steps'])}, "
                    f"optimizer_updates={updates}, elapsed={elapsed:.1f}s"
                )

            checkpoint_dir = run_dir / representation
            checkpoint_dir.mkdir()
            for agent, policy in policies.items():
                torch.save(
                    {
                        "representation": representation,
                        "agent": agent,
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": trainer.optimizers[agent].state_dict(),
                        "optimizer_updates": trainer.optimizer_updates[agent],
                        "config": config.__dict__,
                    },
                    checkpoint_dir / f"{agent}.pt",
                )
    print(f"Pilot artifacts written to: {run_dir.resolve()}")
    return run_dir


def main(argv=None):
    args = _parse_args(argv)
    run_pilot(args)


if __name__ == "__main__":
    main()
