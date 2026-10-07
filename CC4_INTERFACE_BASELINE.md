# CC4 interface baseline

This document freezes the verified environment contract used by the CC4 representation and training code. It describes observable behavior only; it does not claim that any policy has learned a useful defense.

## Scope and provenance

- Project: `cage-challenge-4` working tree.
- Source revision: no Git metadata is available at the project root, so this baseline is tied to the checked-out working tree and should be versioned with the source files.
- Runtime used for validation: Gymnasium 0.28.1, NumPy 1.26.4, PyTorch 2.13.0+cpu, and the `cage-env` interpreter.
- NumPy 2 compatibility remains unverified.
- The standard CC4 topology has nine fixed subnet identities. Host and server counts vary within the generator's supported bounds, and mission phases change connectivity policy during an episode.

## Five-agent interface

CC4 exposes five Blue agents. Agents 0-3 own one local subnet each; agent 4 owns three HQ subnets. Each agent receives its own observation, action mask, and action-index-to-label mapping. A padded action space may give all agents an action-mask shape of `(242,)`, but the labels and action objects at those indices remain agent-specific.

The policy input excludes the trailing 32 received-message bits. It preserves mission phase, subnet identity, blocked-subnet flags, communication-policy context, and candidate-slot process/connection alert flags. The numbered context positions define a tensor layout; their actual subnet meaning must be interpreted with each agent's adapter metadata and one-hot values, because local subnet ownership and row ordering are agent-specific. The 16 slots per local subnet are candidate slots, not confirmed host records. The standard observation does not expose occupancy, and an invalid action may mean either an absent host or a missing Blue session.

The Stage 1 audit found, for seeds 0, 1, 21, 42, and 99 and all five agents, that host-target labels with a valid action-mask entry matched the set of actual hosts on which that Blue agent had a session. This supports an **experimental actionable-host-set contract** defined as host targets with at least one valid host-specific action in the wrapper mask. It does not support a host-presence contract: the actionable set is session- and action-availability-dependent and must not be presented as the set of occupied hosts. The existing fixed-candidate-slot adapter remains unchanged until a separate variable-set adapter is implemented and validated.

## Tick, duration, and reward semantics

For an action selected at environment tick $t$ with duration $d$, the selection tick is the first of the $d$ occupied environment steps. The simulator initializes the action with `remaining_ticks = d`, then decrements it before deciding whether the action executes:

$$
\text{remaining\_ticks}_{t+1} = d - 1.
$$

The simulator emits a forced `Sleep` execution while the decremented remaining count is positive. The selected action executes when that count reaches zero, so the action occupies $d$ environment ticks including the selection tick. For duration 2, the first environment step emits forced `Sleep` and the second environment step executes the selected action. A forced wait is not a new policy decision.

The Blue team reward is shared as a copy across Blue agents at each environment tick:

$$
 r_t = r_t^{(blue\_agent\_0)} = \cdots = r_t^{(blue\_agent\_4)}.
$$

The team return is counted once:

$$
 G = \sum_{t=0}^{T-1} r_t,
$$

not as $5G$. Green work failures, Red effects, action costs, phase-dependent priorities, and disruptive Blue actions can all affect $r_t$. A Green `Sleep` is a no-op for that tick; it does not put a host or Green agent to sleep.

## Step 1 acceptance checks

The contract tests must verify:

1. All five agents parse successfully using their own local adapter and action mapping.
2. Agents 0-3 produce one-subnet features and agent 4 produces three-subnet features.
3. Mission phases 0, 1, and 2 appear during a multi-tick episode.
4. Every environment tick returns one equal shared Blue reward copy per agent.
5. Every selected action is valid under the selected agent's mask and decodes through that agent's labels.
6. A valid duration-2 action causes one forced wait before its action executes.
7. The adapter continues to exclude messages and does not infer host occupancy.

Stage 1 audit command:

```text
python -m pytest CybORG/Tests/test_cc4/test_cc4_host_set_audit.py -q
```

Result: `5 passed`. The simulator host and session data used in this audit are test oracles only and are not supplied to any policy input.

## Stage 2 variable-host adapter status

`CybORG/Agents/CC4VariableObservation.py` now provides a separate experimental adapter. It emits one four-value token per Stage-1-audited actionable host target, using only process-alert, connection-alert, user-role, and server-role values already available through the fixed wrapper. It preserves mission phase, per-agent subnet context, action masks, and action labels, and continues to exclude messages. It does not inspect simulator hosts or sessions.

The Stage 2 tests cover all five agents, one-subnet and three-subnet layouts, sampled legal host populations, message exclusion, role features, and actionable-token/mask consistency:

```text
python -m pytest CybORG/Tests/test_cc4/test_cc4_variable_observation.py -q
```

Result: `8 passed` for the strengthened adapter suite. The shared `VariableCC4InputBuilder` now combines phase one-hot context with subnet context for all variable-host models.

The variable-host comparison models are now defined as follows:

- Variable zero padding: maximum token count is `16 * local_subnet_count`; padded rows carry an explicit padding indicator, and overflow is rejected.
- Variable Deep Sets: shared per-host processing and sum pooling over real actionable-host tokens; an empty host set uses a learned empty pooled representation plus context.
- Variable Transformer: real actionable-host tokens, subnet indices/context, no slot-position embeddings, and an optional padding attention mask for batching.

All three use the same variable model input and independent agent-specific policy heads. The variable representation and rollout suite reports `17 passed`; this is still integration evidence, not an RL result.

These checks validate interface and simulator integration only. They do not validate policy learning, convergence, or performance.

The baseline representation tests use initialized, untrained policies and a short quiet-agent rollout. They therefore support construction, action compatibility, and integration only. They do not establish that the policies learn independently, that the representations differ in performance, that Red/Green conflicts are handled, or that a future learner assigns and discounts rewards correctly across multi-tick actions.
