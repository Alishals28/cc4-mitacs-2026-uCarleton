# Weekly update draft: CC2 Transformer approach in CC4

## Work completed

- Reviewed Andrii Matsevytyi's CC2 paper and traced its Transformer, zero-padding, and Deep Sets implementations in the CC2 clone.
- Inspected the current CC4 generator, wrappers, reward machine, official evaluator, and observation documentation.
- Reviewed the previous student's scenario-generation report and compared its proposed settings with the current checkout.
- Wrote a CC4 implementation and experimental plan in `CC4_TRANSFORMER_IMPLEMENTATION_PLAN.md`.
- Set up a local CC4 smoke-test environment and ran a real five-agent reset and step through `BlueFlatWrapper`.
- Implemented the first policy-input adapter in `CybORG/Agents/CC4LocalObservation.py` and exercised it on all five agents before and after a step.
- Added zero-padding, Deep Sets, and CC2-style Transformer encoders with separate action heads and action mappings for the five Blue agents.
- Added masked categorical/PPO terms, masked DQN bootstrap targets, and a bounded five-agent rollout runner that tracks multi-tick actions and records the shared team reward once per environment tick.

## Findings that shaped the design

- CC4 has five Blue defenders with different local responsibilities and action spaces. The implementation will use five independent policies and no inter-agent message input.
- The official CC4 score is a shared Blue team reward returned to the Blue agents; it must be counted once per environment step, not added five times.
- Host counts and services are randomized at episode generation. The nine-subnet structure is fixed. Mission phases change during an episode and change priorities and connectivity policy.
- The scenario controls in the previous student's report are not in the current CC4 generator. They require edits to the scenario generator and simulation controller. The report's `fixed_min` also uses one user per subnet, below the stock generator's minimum of three. The report itself shows that host counts above the supported maximum break the fixed observation/action representation.
- The main experiment should train and evaluate on the stock randomized CC4 scenario. A legal fixed configuration can help diagnose an interface or learning issue, but should be a temporary curriculum/debugging condition and followed by full randomized training and evaluation.

## Validation status (be precise in the meeting)

**Completed:** source-level design audit; real environment imports, reset, and one step; parsed all five agents' observations and action masks; verified feature shapes (one local subnet for agents 0–3, three for agent 4), omission of the 32 message bits, and rejection of nonzero wrapper padding. The observations were 210-element padded vectors with 242 actions in this checkout. The representation test suite was reported as seven passing tests, including checkpoint round trips. Core imports were updated to Gymnasium and the runtime smoke test passed after removing legacy Gym from the local test environment. The run used NumPy 1.26.4; NumPy 2 remains unverified because package downloads are blocked in this environment.

**Not yet completed:** no PPO/DQN optimizer or training loop, GPU training, official evaluator run, or performance comparison. The new action-masking and rollout tests are authored but not yet executed in this local checkout because its isolated Python environment lacks PyTorch and pytest. One interface limitation was confirmed: the stock observation cannot distinguish an absent host from a present host with no event flags, and the action mask also reflects session availability. Therefore the action mask will only constrain selected actions; it will not be used as a host-presence mask.

## Next implementation checks

1. Run the combined adapter, representation, action-mask, and rollout tests in the Torch-enabled project environment.
2. Verify the rollout's action-duration scheduling, Sleep availability, action-to-label mapping, and one-copy team reward over short seeded episodes.
3. Add a PPO/IPPO training entry point and decide whether to include an observation-only reconstruction auxiliary loss.
4. Run short CPU rollouts and confirm no invalid actions before launching longer GPU training.
5. Train and compare methods on matched randomized seeds; use the official evaluator on held-out seeds after training.

## Suggested spoken update

“I completed the CC2-to-CC4 interface adaptation and implemented the zero-padding, Deep Sets, and Transformer representations as five independent Blue policies with agent-specific action mappings. The adapter excludes message bits and uses only local CC4 inputs; the standard observation does not expose reliable host occupancy, so I do not infer it from the action mask. The representation and checkpoint tests were reported passing. I have now added masked PPO/DQN action utilities and a short five-agent rollout runner that tracks CC4 multi-tick actions and shared reward. Those latest tests still need to run in the Torch-enabled environment. The PPO training loop, GPU runs, and performance comparison are not complete.”
