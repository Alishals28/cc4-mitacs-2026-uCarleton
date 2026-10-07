# Implementation plan: extending Andrii Matsevytyi's CC2 approach to CAGE Challenge 4

## Research question and scope

Can a variable-length, per-host Transformer representation improve the performance and generalization of **five independently trained CC4 defenders** over zero-padded and Deep Sets representations, when all three methods receive the same legitimate local information, use the same RL algorithm and action handling, and are evaluated with the official Blue team reward?

The core comparison is Transformer vs. zero padding vs. Deep Sets. PPO/IPPO-style learning is the primary algorithm because it is better suited to CC4's partial observability, independent multi-agent nonstationarity, delayed action effects, and changing operational priorities. DQN is an optional secondary reproduction baseline motivated by the CC2 paper, not a theoretically preferred CC4 method. Each method comprises five distinct actor/critic or Q-network instances: one for each `blue_agent_0` through `blue_agent_4`. No policy, encoder weights, hidden state, replay buffer, or received messages are shared between agents. A common reward is feedback from the environment, not communication among policies.

No messages are sent by the policies and the 32 received-message bits are removed from all policy inputs. The environment's default empty-message behavior may remain in place. There is no message token, centralized critic, joint observation, or parameter sharing in this study.

## What the source material actually establishes

| Topic | Verified finding | Consequence for this plan |
| --- | --- | --- |
| CC2 method | The six-page paper uses per-host tokens, a Transformer with a `[CLS]` summary, an auxiliary reconstruction loss, and DQN/PPO comparisons against padding and Deep Sets. The CC2 clone contains `TransformerWrapper`, `TransformerStateEncoderV2`, `DeepSetsPermInvEncoder`, `PaddingWrapper`, and `scenario_shuffler.py`. | Reuse the *representation idea* and comparison design; rewrite the integration for CC4 rather than copying the single-agent wrapper. |
| CC2 implementation caveat | `TransformerWrapper.extract_host_state()` reads IP addresses, processes, and ports from `environment_controller.state.hosts`. Its V2 encoder currently comments out positional encoding, trains the reconstruction head inside `forward`, and rebuilds the encoder when the host count changes. | Record these paper/code differences in the reproduction audit. Do not treat all CC2 features as automatically available to a CC4 defender. Keep the CC4 encoder weights intact when host counts change. |
| Five defenders | The CC4 generator assigns one subnet each to agents 0–3, and the three HQ subnets to agent 4; the contractor subnet has no Blue defender. | Build five local models and preserve each agent's distinct observation and action space. |
| Network variation | The official CC4 scenario randomizes 3–10 user hosts and 1–6 servers **per zone at episode creation**. The nine-subnet structure remains fixed. Mission phases alter connectivity during an episode. | Primary claims can address host-count variation across episodes and phase changes within episodes. Mid-episode host joins/leaves or new subnets would require a separate, explicitly modified environment. |
| Standard vector observation | `BlueFlatWrapper` provides mission phase; for each defended subnet, a one-hot subnet identifier, actual blocks, intended communication policy, and one process-alert bit plus one connection-alert bit per potential host; then 32 message bits. Its fixed slots do not distinguish an absent host from an existing host with no alert. | Parse all 16 fixed candidate host slots per local subnet as candidate slots with event flags, not as confirmed host records or host-presence-masked tokens. Exclude the trailing 32 received-message bits while retaining the environment's subnet communication-policy context. Do not invent current security status, full process/port inventories, IP addresses, or actual compromise labels. |
| Action masks and host existence | `BlueFixedActionWrapper` labels a host-target action invalid when the target does not exist **or** the Blue agent lacks a session on it. A Stage 1 audit across seeds 0, 1, 21, 42, and 99 found that valid host-target labels matched actual hosts with a Blue session for all five agents. | The standard fixed-slot representation still must not use the mask as a host-presence mask. A separate experimental variable-set adapter may define an **actionable host set** as host targets with at least one valid host-specific action in the wrapper mask. This set is session/action-availability-dependent, not confirmed occupancy. Keep simulator host/session data test-only and validate the new adapter separately. |
| Reward and score | `BlueRewardMachine` computes one phase-dependent Blue team penalty from green work/service failures and successful Red impact, plus the simulator's action cost. `get_reward(agent)` returns the Blue team's reward for every Blue agent. The official evaluator sums the **mean** of the five returned Blue rewards each tick over 500 ticks, then averages 100 episodes. | Train each independent learner on the *single* shared team reward returned by the wrapper. Report the official team episode score once; do not sum the five copies and multiply it by five. |
| Previous student's scenario controls | The attached 24-page PDF proposes generator parameters for fixed/randomized host counts, minimum/maximum services, and progressive/fixed/random phases. It describes replacing both `EnterpriseScenarioGenerator.py` and `SimulationController.py`; these parameters are **not** present in the current CC4 checkout's `EnterpriseScenarioGenerator.__init__`. The document also tests host counts below the official user-host minimum and warns that counts above the maxima break the fixed spaces. | Treat the PDF as a separate experimental modification, not an already-supported switch. Do not replace core CC4 files for the primary study. If controlled scenarios are needed for debugging, port only the required option behind an explicit config and test it against the current checkout. Keep all counts within the wrapper's supported maximum. |

The local source anchors are [`CC4 README`](README.md), [`BlueFlatWrapper`](CybORG/Agents/Wrappers/BlueFlatWrapper.py), [`BlueFixedActionWrapper`](CybORG/Agents/Wrappers/BlueFixedActionWrapper.py), [`EnterpriseScenarioGenerator`](CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py), [`BlueRewardMachine`](CybORG/Shared/BlueRewardMachine.py), and [`official evaluator`](CybORG/Evaluation/evaluation.py). The CC2 anchors are [`TransformerWrapper`](../cc2-mitacs-2025-uCarleton/CybORG/CybORG/Agents/Wrappers/TransformerWrapper.py) and [`TransformerStateEncoderV2`](../cc2-mitacs-2025-uCarleton/CybORG/CybORG/Agents/Wrappers/TransformerStateEncoderV2.py). The attached “Implementation Plan - Alisha.pdf” proposes Idea 1; its message-token element is excluded here on the professor's instruction.

## Model and interface design

### 1. A common, observable input contract

Build one adapter around the official `EnterpriseMAE`/`BlueFlatWrapper` and `BlueFixedActionWrapper` outputs. At every decision, it gives a policy only its own Blue observation, its own action mask and labels, and static metadata needed to interpret its own host slots. It strips the message block. It must not expose other agents' observations, raw simulator `State`, Red or Green internal state, reward breakdown before acting, or a true attack label. Add a contract test that logs the source of every feature and proves that the adapter works through the same `Submission.wrap` interface used by official evaluation.

For every fixed candidate host slot (16 per local subnet), construct one token from: process-alert flag, connection-alert flag, host role (user/server), local host slot, and subnet identity. For agent 4, preserve which of the three HQ subnets owns each slot. These are fixed candidate host slots with event flags, not confirmed host records. The stock observation does not expose host existence, and the action mask conflates absence with lack of a Blue session; therefore all candidate slots remain in the representation and no host-presence-masked token is claimed. A missing host and a present host with no reported events may be observationally identical. Carry mission phase and the owned subnet's actual block and intended-policy rows as explicit context features. These are local policy inputs already present in the official wrapper.

An `Analyse` result or other information from raw observations may form a **separate richer-observation experiment** only after an audit establishes that it is genuinely delivered to that Blue agent. If included, extend the common input contract for all three methods at once. Do not use the CC2 wrapper's direct access to all live hosts as a shortcut. The standard-vector experiment is the primary result.

The Stage 2 experimental variable-host adapter is a separate input path from the fixed candidate-slot adapter. It emits one token per actionable host target established by the Stage 1 mask audit, not one token per confirmed occupied host. It uses only wrapper-visible alert and role features plus local context. The variable comparison now has three model paths consuming one shared builder: zero padding with maximum `16 * local_subnet_count` tokens and an explicit padding indicator, Deep Sets over real tokens with defined empty-set behavior, and a Transformer over real tokens without slot-position embeddings. The existing fixed-slot models remain separate from this variable-host comparison.

### 2. Three representations with the same information

**Transformer:** Project each candidate host slot's CC4 feature vector to an embedding. Add role, subnet, and stable host-slot embeddings; prepend a learned `[CLS]` token. Supply the agent's mission/policy/block context through a context projection or context token. Use a two-layer self-attention encoder as in the CC2 paper. Since the official observation does not reveal host presence, do not mask tokens based on hidden state or the action mask. The `[CLS]` output feeds that agent's policy and value/Q heads. Use the same encoder parameters for every episode within that agent.

**Deep Sets:** Use the same candidate-slot features and embedding width, process all candidate slots with a shared `phi` network, aggregate with a permutation-invariant pool, and combine with the same context vector. Include a local-plus-global equivariant layer to match the CC2 comparison. Do not mask slots using unavailable host-presence information.

**Zero padding:** Lay out those same candidate-slot features in stable local host slots up to CC4's documented maximum (16 per owned subnet, 48 for agent 4), then concatenate the same context before an MLP. Slots beyond the observed event flags are not treated as known absent hosts. Padding is a representation baseline, not a different observation source. Each agent keeps its own input and output dimensions.

Match approximate model parameter counts and training budgets; report remaining differences. Freeze the shared feature schema and action mapping before training. A Transformer advantage is only attributable to representation if the baselines receive identical observables and action handling.

### 3. Local auxiliary reconstruction

Retain Andrii's auxiliary objective only if it can be defined against the same observable fixed candidate-slot representation for every baseline. A decoder can predict per-slot alert features; it must not reconstruct true host presence or hidden host state. Keep the target detached when reconstructing learned token embeddings; also report raw alert-feature reconstruction so a low loss cannot be explained entirely by a collapsed embedding. Weight the reconstruction term as a documented hyperparameter selected on validation seeds. Train it together with PPO/DQN using one optimizer schedule, and save it with the encoder; turn it off during evaluation.

The CC4 alert bits are sparse, so run a Transformer-without-reconstruction ablation. If reconstruction does not help, report that result; do not claim that the paper's auxiliary loss transfers unchanged. No global or unobserved-host reconstruction target is permitted.

### 4. Action selection and independent learning

Use the existing fixed action index list for each agent. PPO masks invalid logits before sampling; DQN masks invalid actions in epsilon-greedy selection **and** in the bootstrap max/target computation. At every reset and step, assert that chosen actions are valid in the official action mask and map back to the intended local target. The different action durations remain part of the environment; do not assume each agent starts a new command every tick. Buffer only transitions for active agents and follow the wrapper's termination/truncation semantics.

Train five independent PPO policies (IPPO in the limited sense of separate local actors and critics with a shared team reward). Each learner observes only its own input and previous local history if an explicit recurrent ablation is later added. Independent policies may be trained while all five act in the same simulated episode; this is necessary to observe the correct team consequences of their actions. For the primary comparison, use the same PPO optimizer, discount, rollout budget, reward, valid-action mask, schedule, and seed set across the three representations. Do not reuse Andrii's CC2 hyperparameters without checking their effect in CC4's 500-step, phase-changing setting.

### 5. Feature ablations and channelization

The current CC4 representation combines four candidate-slot values through one `Linear(4 -> hidden)` projection: malicious-process alert, network-connection alert, user-role metadata, and server-role metadata. These are CC4-specific observable features and must not be described as direct equivalents of the four streams in Andrii's paper, which describe security status, processes, ports, and IP address. The paper and the CC2 V2 code path must also be reported separately because the code path does not necessarily concatenate every calculated feature stream into the final token.

Keep mission phase and subnet context unchanged, keep model dimensions unchanged, and evaluate these input variants:

| Variant | Process alerts | Connection alerts | Role metadata | Context |
| --- | --- | --- | --- | --- |
| Process-alert ablation | Off | On | On | On |
| Connection-alert ablation | On | Off | On | On |
| Role-metadata ablation | On | On | Off | On |
| Full-input control | On | On | On | On |

The feature-selection pilot may be Transformer-only when compute is limited, but that must be reported as a Transformer-only development ablation. The final three-way comparison must give zero padding, Deep Sets, and Transformer the same frozen feature variant, seeds, PPO settings, reward, action masks, and transition budget.

Separately, compare the current mixed projection with separate learned projections for alert features and role metadata. This is a channelized-architecture ablation, not a feature-usefulness ablation. If used in the representation comparison, the same channelized input builder must feed all three encoders.

### 6. Dynamic conflicts, action duration, and reward credit

CC4 does not coordinate agents by waiting for one another. Every action is attempted against the simulator state at the time it executes. A Green `Sleep` is a no-op for that tick; it does not put a host or agent to sleep. Green work can fail when a Blue action has made the host unavailable or its services inactive, and the resulting failure can reduce the shared Blue team reward. Red actions and other agents can also change the state before a Blue action completes.

Action durations are part of the environment dynamics. Once a multi-tick action starts, it cannot be cancelled; while it runs, other agents continue acting. A forced wait during that interval is not a new policy decision and must not be recorded as one. The rollout and training data must distinguish:

- a policy decision and its selected action;
- forced waiting while that action remains in progress;
- action success, failure, or completion status when observable;
- the reward received at every environment tick; and
- the active-agent, terminated, and truncated status at that tick.

The PPO transition/rollout design must preserve the full action duration and accumulate or otherwise correctly account for rewards received while the action is running. It must not assume that every environment tick produces a fresh decision. The shared reward is a common learning signal, not causal attribution: it does not identify which other agent or event caused a state change.

Training and evaluation must use the dynamic environment, including Green activity, Red interference, action failures, delayed effects, and mission-phase changes. Matched methods must use the same scenario seeds so conflict consequences are comparable. Report invalid-action rate, action outcome counts, forced-wait counts, reward by phase, and reward by action-duration category in addition to team return.

### 7. Algorithm scope and assumptions

PPO is the primary algorithm for the CC4 study. Independent PPO policies still face a nonstationary multi-agent environment and partial observability, so the implementation must describe the method as independent decentralized control with a shared team reward, not as a converged centralized solution. A recurrent or centralized-critic extension is outside the first implementation unless explicitly added as a separate experiment.

DQN may remain as an optional secondary baseline to compare with the CC2 paper, but its assumptions must be documented: independent learners change one another's transition distributions, local observations are not guaranteed to be Markov, and epsilon-greedy exploration does not remove nonstationarity or sparse-reward difficulties. If DQN is implemented, use the tested invalid-action mask in both epsilon-greedy selection and target maximization, reject all-invalid masks for nonterminal states, and report it as exploratory rather than as the preferred CC4 method. If time or compute is limited, retain the DQN masking utilities and omit the full DQN trainer rather than implying a complete comparison.

## Execution sequence and acceptance gates

1. **Freeze provenance and interfaces.** Record source revisions and dependency state for both clones. Capture observations, action labels/masks, active-agent timing, reward copies, and action durations across multiple seeds and mission phases. Keep the current Gymnasium/NumPy 1.26.4 validation record and separately track NumPy 2 compatibility.
2. **Lock the common observable contract.** Preserve mission phase, subnet context, candidate-slot alert features, role metadata, and action masks. Treat numbered subnet-context positions as a feature layout whose actual subnet meaning comes from each agent's adapter metadata and one-hot values; do not imply that one position has the same local ownership meaning for every agent. Verify that received messages do not affect policy features, that host occupancy is not inferred, and that every action index is decoded through the correct agent's mapping.
3. **Validate representation baselines.** Maintain zero-padding and Deep Sets baselines with identical features, context, action handling, model-output sizes, checkpoint behavior, and independent per-agent parameters. The acceptance claim is limited to separate construction, action compatibility, and short rollout integration with initialized policies; it is not evidence of independent learning, comparative performance, realistic conflict handling, or duration-aware reward credit.
4. **Validate the Transformer.** Maintain the two-layer `[CLS]` encoder, test all-zero candidate slots, the fixed 16 candidate-slot bound, and reuse across CC4's one-subnet and three-subnet local inputs. Do not describe this as arbitrary occupied-host-count scaling because the standard vector wrapper does not expose host occupancy or variable token counts. Add the channelized-projection ablation only as a matched architectural experiment.
5. **Add feature ablations.** Run the three feature-removal variants and full-input control on development seeds. Freeze the selected feature contract before the three-way representation comparison, or explicitly report the ablation as Transformer-only.
6. **Build the PPO training path first.** Implement five independent actor-critic policies with shared team reward, action masks in sampling and update distributions, all-invalid-mask checks, and duration-aware rollout records. Preserve policy decisions separately from forced waits and collect reward every environment tick.
7. **Run a short dynamic PPO integration pilot.** Use Green and Red activity, phase changes, multiple action durations, and matched seeds. Verify action outcome records, delayed effects, reward accumulation, active-agent timing, invalid-action rate, and shared-reward accounting before long training.
8. **Decide on DQN explicitly.** If resources permit, implement it only as a secondary masked baseline with documented nonstationarity and partial-observability limitations. Otherwise retain the tested DQN target utilities and report the DQN comparison as out of scope.
9. **Run matched experiments.** Use common seeds, transition budgets, PPO settings, action masks, and reward handling across the three representations. Do not claim a representation or algorithm improvement from reset/step, shape, or short-rollout checks.
10. **Evaluate and package evidence.** Use held-out seeds and the official evaluation protocol when feasible. Report team return, uncertainty, invalid actions, action outcomes, forced waits, reward by phase and duration, host-count bins, runtime, and parameters. Preserve checkpoints, configurations, and commands needed to reproduce each result.

## Static scenarios and training recommendation

The attached scenario-modification report is useful as a *controlled diagnostic idea*, not evidence that a fixed environment is required for learning. In the stock CC4 code, host counts and services are randomized when a new episode is generated; subnet membership/count does not change mid-episode. The mission phase progresses during the 500-step episode and changes both policy context and reward priorities. Therefore training on one fixed topology/phase can help isolate an interface or reward bug, but by itself does not train robustness to the distribution that evaluation uses.

Recommended sequence:

1. For interface debugging, use a fixed seed and short episodes to make failures reproducible; this is still a deterministic random stream and must not be described as an identical regenerated scenario unless verified from the reset behavior.
2. If the policy cannot learn basic action/observation handling, use a small, controlled curriculum as a diagnostic: fixed legal host counts, keep service count randomized initially, and keep the phase progressive. If a fixed phase is used to isolate phase-specific behavior, test all three phases, then restore progressive phases.
3. Train the reported model with stock randomized host counts/services and progressive phases across a fixed, documented training seed list. Hold out seeds for evaluation. This exposes it to legal scenario variation and phase changes while retaining reproducibility.
4. Evaluate on fresh seeds with the unmodified official generator and official 500-step protocol. Report any fixed-scenario curriculum as a training stage; never use its fixed scenario as the sole result or as a substitute for the official dynamic distribution.

The attached PDF's `fixed_min` example specifies one user per subnet, below the current CC4 generator's `MIN_USER_HOSTS = 3`; it is a useful test of wrapper tolerance but is outside the official scenario distribution. Its over-maximum tests are expressly invalid for standard observation/action spaces. Do not make either condition part of the primary training/evaluation claim.

The first local runtime smoke test passed: five Blue agents reset and took one step through `BlueFlatWrapper`, and the new local-observation adapter parsed all five observations and current action masks before and after the step. A recorded run of the existing CC4 pytest suite in `cage-env` completed with 680 passed and 15 skipped. That result does not validate the new adapter-specific pytest file, which has not yet been run in the isolated environment because pytest is unavailable there. The core package imports were moved from legacy Gym to Gymnasium because the checked-in requirements comment out Gym. The runtime check used Gymnasium 0.28.1 with NumPy 1.26.4; NumPy 2 compatibility remains unverified. Official-evaluator integration, training, and performance remain unverified. Package downloads were blocked during the NumPy 2 check.

## Main validity risks and how the design addresses them

- **Information leakage:** CC2's implementation reads internal host state. CC4's primary experiment uses only official per-agent observations and masks. Raw-observation extensions require a separate audit and matched baselines.
- **False topology claim:** The default CC4 graph of subnets is fixed; host counts vary per episode and communication policy varies by phase. This study must not claim arbitrary new subnet support or within-episode host churn.
- **Topology masking:** A physical firewall closure does not imply that two locally observed host tokens should be forbidden from attending to each other. Start with phase/policy/block as features; treat a phase-derived attention mask as a later ablation only if it models an explicit, tested information constraint.
- **Action mismatch:** Agents 0–3 and agent 4 have different action-space lengths and meanings at the same padded index. Five separate heads and local action maps avoid this issue.
- **Reward multiplication:** All five Blue agents receive the same team reward dictionary. One step has one team reward, not five separate rewards to be summed.
- **Overclaiming collaboration:** The policies act in a shared world and optimize a shared score, but exchange no observations or messages. Describe the method as independent decentralized control with a common team reward.

## Prior CC4 solutions and what they teach this study

The [official CC4 analysis](https://ojs.aaai.org/index.php/AAAI/article/download/35158/37313) reports that leading submitted heuristics outperformed the leading MARL agent in its standard evaluation and identifies invalid actions on absent hosts as a common failure. Therefore action masking is essential, but it must use the official action-availability mask and must not be described as host-presence masking. [PUNCH's published submissions](https://github.com/PUNCH-Cyber/cage-4-submissions) show why `Analyse` results and action-index alignment matter: their heuristic exposed observed file information through a custom wrapper, whereas their shared-policy PPO had to realign padded actions. This study keeps per-agent policies and gives any optional `Analyse` feature to every representation. [Cybermonic's KEEP submission](https://github.com/cybermonic/cage-4-submission) demonstrates a viable five-policy independent PPO training loop with local observations and target-aware actions, but uses a graph encoder; its design informs rollout and action handling, rather than replacing the requested Transformer comparison. These reported competition scores are context, not directly comparable to a new method until evaluation versions, seeds, and wrappers are matched.

## Sources

- Matsevytyi et al., *Scalable RL for Autonomous Cyber Defense in Varying Networks: A Transformer-based Approach*, attached `F:\Mitacs Research\Andrii Paper.pdf`.
- Alisha, *Implementation Plan - Alisha*, attached `F:\Mitacs Research\Implementation Plan - Alisha.pdf`, Idea 1.
- [Official CC4 challenge details](https://github.com/cage-challenge/cage-challenge-4/blob/main/README.md), [official documentation](https://cage-challenge.github.io/cage-challenge-4/pages/reference/reference/), and checked-out CC4 source files linked above.
- [CC4 competition analysis](https://ojs.aaai.org/index.php/AAAI/article/download/35158/37313), [PUNCH submission](https://github.com/PUNCH-Cyber/cage-4-submissions), [Cybermonic submission](https://github.com/cybermonic/cage-4-submission).
