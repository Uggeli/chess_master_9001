# Model Proposition: Memory-Assisted Bayesian Chess Model

## Goal

Build a chess model that does more than predict a move in one shot.

Target behavior:

- encode the current board into a latent state
- retrieve similar past states from Hopfield memory
- predict likely opponent replies
- evaluate short future branches
- choose the move with the best expected outcome

This is not Monte Carlo Tree Search. It is a memory-assisted, belief-based planner.

Current repository status:
- NOTHING

---

## Core Idea

The model should combine three capabilities:

1. **Perception**
   - understand the current board state
2. **Memory**
   - recall similar situations and their outcomes
3. **Forward prediction**
   - estimate what the opponent is likely to do next and what happens after that

The desired decision rule is:

$$
a^* = \arg\max_a \sum_b P(b \mid s, a, m) \cdot V(s')
$$

Where:

- $s$ = current state
- $a$ = our candidate move
- $b$ = opponent reply
- $m$ = retrieved memory context
- $s'$ = resulting state after $a$ and $b$
- $V(s')$ = predicted value of that resulting state

So the model does not only ask:

- "what move looks good now?"

It asks:

- "if I do this, what will they probably do, and how good is that future for me?"

The memory system should be explicitly **dual-scale**:

- **long-term memory** = persistent positional knowledge across games
- **short-term memory** = game-local trajectory and opponent behavior within the current game

This separation is important because these two memory roles are different:

- long-term memory answers: "what kinds of positions are these?"
- short-term memory answers: "how did this particular game get here, and what is this opponent currently revealing?"

---

## Proposed Architecture

### 0. Dual-Memory Structure

The full system should use both persistent and game-local memory.

#### Long-term memory

- persistent across games
- curated and bounded
- retrieved by positional similarity
- acts as analogical prior

Examples:

- opening preparation
- structural pattern knowledge
- recurring tactical and strategic motifs

#### Short-term memory

- reset at the end of each game
- stores the recent local trajectory
- can be implemented as a ring buffer or explicit sequence window
- does not need Hopfield retrieval initially

Examples:

- recent moves
- recent positions
- recent evaluation deltas
- opponent tendencies shown in the current game

This gives the model three distinct sources of signal:

- current board state
- current-game trajectory
- historical analogies from past games

Recommended first implementation:

- short-term memory as the last `N` game events encoded as sequence tokens
- long-term memory as the existing Hopfield-style store

The intended retrieval flow is:

$$
q_t = f(s_t, \text{short-term context})
$$

$$
m_t = \text{HopfieldRetrieve}(q_t, \text{long-term store})
$$

So short-term context shapes which long-term analogies are retrieved.

---

### 1. State Encoder

Input:

- board tensor
- optional move history
- short-term game context
- side to move

Output:

- latent state `z_t`
- spatial tokens for transformer processing

Current base:

- [chess_engine/encoder.py](chess_engine/encoder.py)
- [chess_engine/model.py](chess_engine/model.py)

Proposed role:

- produce a board representation that is stable, discriminative, and suitable for both memory lookup and forward prediction
- incorporate recent game trajectory before long-term retrieval happens

Important refinement:

- the encoder likely needs **two projection heads** on top of a shared backbone
   - one head for policy/value computation
   - one head for memory retrieval

Reason:

- the embedding space that is best for move prediction is not necessarily the embedding space that is best for analogical retrieval
- separating them reduces the risk that retrieval collapses because policy gradients distort the memory geometry

Short-term context recommendation:

- do not feed only raw past positions
- include recent moves and evaluation deltas

This allows the model to represent:

- trajectory awareness
- momentum
- whether a plan is improving or deteriorating
- whether the opponent is behaving aggressively, passively, or unusually in the current game

Minimal short-term event schema:

- `state_emb_t`
- `move_idx_t`
- `side_to_move_t`
- `eval_t`
- `eval_delta_t`

Important detail:

- evaluation signals must use a **consistent perspective convention**
- do not mix "side-to-move" and "white-centric" values in the same buffer

Recommended rule:

- store all short-term values from a single fixed perspective per training example
- simplest choice: always convert to the current decision-maker's perspective at that timestep

Otherwise, `eval_delta_t` becomes noisy and the model cannot reliably interpret momentum.

---

### 2. Hopfield Episodic Memory

Input:

- current latent state `z_t`

Output:

- retrieved similar episodes
- aggregated memory context `m_t`
- confidence / familiarity score

Current base:

- [chess_engine/hopfield.py](chess_engine/hopfield.py)
- [chess_engine/memory.py](chess_engine/memory.py)

Long-term target memory items can become richer episodes:

- `state_emb`
- `side_to_move`
- `action`
- `next_state_emb`
- `opponent_reply`
- `reply_state_emb`
- `value`
- `game_outcome`
- `confidence`
- `source` (`stockfish`, self-play, correction)

However, the **initial implementation should stay smaller**. Start with:

- `state_emb`
- `move_idx`
- `value`
- `outcome`
- `source`
- `importance`

Only add opponent-reply fields once opponent modeling is working.

Memory should act as a **prior**, not as the sole decision maker.

Crucially, the query into long-term memory should not be based on the board alone.
It should be based on the board embedding **after it has been enriched by short-term game context**.

This enables behavior like:

- "this position looks structurally familiar, but the recent trajectory is non-standard"
- "the usual plan for this structure may not apply because the current game has already diverged"

---

### 3. Our Policy Head

Predict candidate moves from the current state.

$$
\pi(a \mid s, m)
$$

This head should use both:

- current transformer state
- retrieved memory context

Purpose:

- produce top-k candidate moves for explicit short-horizon evaluation

Current base:

- policy head in [chess_engine/model.py](chess_engine/model.py)

---

### 4. Opponent Reply Model

After simulating one of our candidate moves, estimate the opponent's next move distribution.

$$
P(b \mid s, a, m)
$$

This can be implemented in two stages:

#### Version A: reuse the same model

- apply our candidate move on a copied board
- run the model on the resulting board from opponent-to-move perspective
- interpret policy output as opponent reply distribution

Pros:

- very simple
- immediately available from the current codebase
- good enough as a first validation path

Risk:

- it introduces a mirror bias in self-play, because the model predicts replies in its own style rather than a clean adversarial model

#### Version B: explicit opponent head

- separate head trained specifically to predict opponent replies from afterstates

Pros:

- cleaner separation of concerns
- easier to train as a true adversarial predictor
- avoids overfitting opponent predictions to our own policy style

Recommendation:

- start with Version A for speed
- design the inference interface so Version B can replace it without changing planner logic

The opponent model is the key piece that turns the system from:

- static evaluation

to:

- anticipatory decision-making

---

### 5. Afterstate / Transition Model

Optional but recommended.

Learn a latent transition:

$$
\hat z_{t+1} = f(z_t, a_t)
$$

And optionally:

$$
\hat z_{t+2} = f(\hat z_{t+1}, b_t)
$$

This allows planning in latent space rather than always running full board re-encoding.

Two options:

#### Board-exact version

- use real chess move application with `python-chess`
- re-encode resulting boards exactly
- simpler and safer initially

#### Latent dynamics version

- predict successor embeddings directly
- faster and more flexible later
- harder to stabilize

Recommended order:

1. board-exact first
2. latent dynamics later

---

### 6. Bayesian Belief Merger

Combine:

- learned dynamics/value estimate
- memory-based prior from similar states
- opponent reply distribution

Conceptually:

$$
p_{combined} \propto p_{model}^{\alpha} \cdot p_{memory}^{1-\alpha}
$$

Where $\alpha$ controls trust in direct model prediction vs memory prior.

Initial recommendation:

- **do not learn $\alpha$ first**
- derive it from retrieval quality using a hard-coded confidence proxy

Suggested proxy:

- top retrieval cosine similarity
- or gap between top-1 and top-k retrieved similarities

Example rule:

- if best similarity is below threshold, set $\alpha \approx 1.0$
- if best similarity is high, lower $\alpha$ and allow memory to influence the prediction more

Only later, once retrieval and opponent modeling are stable, consider a learned confidence head.

If memory is confident and highly familiar:

- trust it more

If the state is novel:

- trust direct model prediction more

This is the part that makes the system "Bayesian-like" in behavior, but it should begin with fixed heuristics before any learned merger.

---

### 7. Short-Horizon Planner

Use explicit shallow rollout over moves.

Recommended first version:

1. generate top-k candidate moves from `\pi(a \mid s, m)`
2. for each candidate move, apply it to a board copy
3. predict top-r opponent replies
4. evaluate resulting states
5. aggregate expected value
6. choose best move

Decision rule:

$$
Q(s,a) = \sum_{b \in \text{TopR}} P(b \mid s,a,m) \cdot V(s_{ab})
$$

Then:

$$
a^* = \arg\max_a Q(s,a)
$$

This gives explicit forward thinking without full MCTS.

Short-term memory should also be updated inside this loop so the planner can reason about:

- continuity of plans
- whether sacrifices are being justified
- whether the opponent is consistently choosing certain reply types

### 8. Recursive Single-Line Projection

Before introducing any beam search or branching planner, the system should support a much simpler mechanism:

- choose our best move
- predict the opponent's most likely reply
- apply both moves
- repeat for a fixed horizon

This produces a **single projected game trajectory** rather than a search tree.

Formally, if the model is used recursively for both sides, the projected line is:

$$
s_t \rightarrow a_t \rightarrow b_t \rightarrow a_{t+1} \rightarrow b_{t+1} \rightarrow \dots
$$

for a horizon such as 20 moves per side.

This is not intended as an exact forecast. It is intended as a **comparative planning tool**.

This framing should govern implementation decisions:

- the projection does not need to be literally accurate
- it needs to be useful for **ranking root candidates consistently**
- relative quality matters more than exact move-by-move correctness

The important question is not:

- "is this exact 20-move line what will happen?"

The important question is:

- "does candidate move A lead to a better projected future than candidate move B under the model's own continuation dynamics?"

Recommended initial rule:

1. generate top-k root candidate moves
2. for each candidate, force that move on a board copy
3. recursively alternate:
   - our best predicted move
   - opponent's most likely predicted reply
4. continue for a fixed horizon
5. score the projected line using:
   - endpoint value
   - value trend over time
   - agreement or disagreement with the simpler two-ply planner
   - optional hand-crafted trajectory features

This gives a cheap non-branching planner.

Computationally, this is practical:

- one projection of 20 full moves is about 40 forward passes
- with `k = 5` root candidates, inference remains manageable

Why this is useful:

- plans emerge from the projected line instead of being explicitly represented
- the model can compare strategic futures, not just current positions
- short-term memory can store the projected continuation and preserve plan continuity on later real moves

Additional scoring signal:

- compare the projection ranking against the simpler two-ply ranking
- agreement is a useful confirmation signal
- disagreement is informative and should be tracked explicitly

Interpretation of disagreement:

- the short tactical window may be seeing something the long projection misses
- or the longer projection may be capturing strategic consequences the two-ply evaluator cannot see

In early versions, disagreement should trigger caution rather than blind trust in the deeper projection.

Important limitation:

- the projection is a self-consistent line under the model's own beliefs
- it may amplify opponent-model mirror bias
- an error early in the line can make later projected moves fictional

Because of that, single-line projection should first be used for **relative ranking of root candidates**, not as a literal prediction of the next 20 moves of the real game.

### 9. Adaptive Projection Horizon

The projected trajectory length should not be fixed. It should depend on how confident the model is in the continuation.

Core idea:

- if the model is confident, project farther
- if uncertainty is high, stop early
- if confidence collapses during the rollout, terminate the projection at that point

Conceptually:

$$
H = g(c_{policy}, c_{memory}, c_{value}, c_{opp})
$$

Where:

- $H$ = rollout horizon
- $c_{policy}$ = confidence from policy sharpness
- $c_{memory}$ = confidence from retrieval similarity / familiarity
- $c_{value}$ = confidence from value stability
- $c_{opp}$ = confidence from opponent-reply certainty

One practical approximation:

$$
c = w_1 c_{policy} + w_2 c_{memory} + w_3 c_{value} + w_4 c_{opp}
$$

$$
H = H_{min} + \left\lfloor c \cdot (H_{max} - H_{min}) \right\rfloor
$$

Recommended starting values:

- $H_{min} = 2$
- $H_{max} = 20$

### Candidate confidence signals

#### Policy confidence

- top-1 policy probability
- margin between top-1 and top-2 moves
- policy entropy

#### Memory confidence

- top retrieval cosine similarity
- average top-k retrieval similarity
- gap between top-1 and top-k retrieved items

#### Value confidence

- smoothness of projected value over the last few plies
- agreement between direct evaluation and memory-conditioned evaluation

#### Opponent-model confidence

- entropy of predicted opponent reply distribution
- margin between top opponent replies

### Early stopping during rollout

Even if the root state suggests a long projection, rollout should stop early if confidence deteriorates.

Suggested stopping triggers:

- retrieval similarity falls below threshold
- opponent entropy rises above threshold
- projected value oscillates sharply over consecutive plies
- model and memory-conditioned estimates diverge too much

This gives an **adaptive-depth Bayesian rollout**:

- deeper in familiar, coherent situations
- shallower in tactical, unstable, or novel situations

This is preferable to forcing every projected line to 20 moves because it reduces projection fiction and saves compute.

Implementation note:

- this is a **target design**, not the first implementation target
- the project should first validate fixed-horizon projection before introducing adaptive-depth control
- if a fixed horizon already gives consistent gains, adaptive depth may be unnecessary for an early version

---

## Proposed Data Stored in Memory

Each memory entry should represent a small episode, not just a static lookup point.

### Minimal starting schema

- `state_emb`
- `move_idx`
- `value`
- `outcome`
- `source`
- `importance`

This is the recommended starting point because it keeps the write path simple while validating whether retrieval helps at all.

### Short-term game buffer schema

Short-term memory should be separate from the long-term store.

Suggested starting fields:

- `state_emb`
- `move_idx`
- `side_to_move`
- `value`
- `value_delta`

Perspective rule:

- `value` and `value_delta` must be normalized to a consistent viewpoint
- recommended: store them from the acting side's perspective for each event
- alternative: always store from White's perspective, but then keep that convention everywhere

Optional later additions:

- local tactical flags
- legality-pressure summary
- opponent-style summary token

### Expanded future schema

- `state_fen`
- `state_emb`
- `move_idx`
- `move_prob`
- `value_before`
- `next_fen`
- `next_state_emb`
- `opponent_reply_idx`
- `reply_fen`
- `reply_state_emb`
- `value_after_reply`
- `outcome`
- `importance`
- `timestamp`
- `source`

The expanded schema should only be introduced after the minimal schema and opponent reply modeling are already stable.

### Memory growth and curation

Memory curation must be explicit. A useful system needs forgetting and consolidation, not unbounded accumulation.

Recommended strategy:

- keep a bounded store
- use importance-weighted eviction
- decay stale items over time
- preserve rare but high-value tactical patterns
- optionally consolidate clusters of near-duplicate states into prototypes

The existence of short-term memory makes long-term curation easier:

- long-term memory does not need to preserve game-local continuity
- short-term memory already carries local plans and recent behavioral evidence
- this means long-term storage can prune more aggressively and focus on reusable positional knowledge

Possible importance signals:

- retrieval frequency
- correction usefulness
- surprise / prediction error
- game outcome contribution
- Stockfish disagreement magnitude

This would let memory answer questions like:

- what usually happened in similar positions?
- what reply did the opponent tend to make?
- which continuations ended well?
- how trustworthy is this analogy?

---

## Training Plan

### Phase A: Strong one-step predictor

Train:

- policy head
- value head
- memory retrieval quality

Supervision sources:

- Stockfish move
- Stockfish evaluation
- retrieval target on similar states

Goal:

- get a stable and discriminative encoder
- validate that policy/value quality is strong enough to support shallow planning

---

### Phase B: Opponent reply prediction

Create training tuples:

- current position `s`
- our move `a`
- opponent move `b`

Train:

$$
P(b \mid s, a)
$$

Possible data sources:

- Stockfish best replies
- self-play replies
- human game continuations

Goal:

- model the adversary explicitly

---

### Phase C: Two-ply expected-value training

Train the model to score candidate moves by expected future value:

$$
Q(s,a) = \sum_b P(b \mid s,a) V(s_{ab})
$$

Losses:

- policy imitation loss
- value loss
- opponent reply prediction loss
- retrieval alignment loss
- optional consistency loss between predicted and actual future embeddings

Validation gate:

- if two-ply expected scoring does not beat one-shot move selection with the current value head, the project should pause here and improve the one-step model before adding more memory complexity

---

### Phase D: Memory confidence learning

Only after the simpler merger is stable.

Train a confidence head to estimate whether retrieved memory should be trusted.

Example targets:

- retrieval usefulness
- novelty of current state
- mismatch between memory prior and actual continuation

Goal:

- avoid overtrusting weak analogies

---

## Recommended Incremental Implementation

### Step 1 — Stabilize current memory model

Keep current stack, but improve memory semantics:

- make memory entries checkpoint- or run-specific
- keep the schema minimal at first
- add explicit memory curation and forgetting rules
- measure whether retrieval changes move choice

### Step 1.5 — Add short-term game memory

Before adding a richer long-term schema, add a lightweight current-game buffer:

- keep the last `N` game events
- encode moves, positions, and value deltas
- feed them into the encoder as additional context tokens

This should be the first step toward non-stateless evaluation.

### Step 2 — Add opponent-aware move selection

In `select_move()`:

- generate top-k moves
- simulate each move
- predict top-r opponent replies
- evaluate expected value

This is the simplest path to forward thinking and should be treated as the main validation gate for the entire proposal.

### Step 2.5 — Add recursive projected trajectory scoring

Without changing the architecture, add a loop that projects one continuation line forward for a fixed horizon.

Recommended procedure:

- top-k root candidates
- for each root candidate, roll out one predicted line for a fixed horizon such as 10 full moves
- score the endpoint and trajectory trend
- choose the root move with the strongest projected continuation

This is the lowest-complexity route to deeper strategic comparison.

### Step 2.6 — Make projection horizon adaptive

Only after fixed-length single-line rollout has been shown to improve root move selection.

After that, replace it with confidence-controlled rollout depth if there is evidence that fixed depth is leaving performance on the table.

Procedure:

- compute rollout confidence at the root
- set an initial horizon bound
- recompute confidence during the rollout
- stop early if uncertainty rises too much

This should be introduced before any wider branching planner.

### Step 3 — Add explicit opponent head

Extend the model with a dedicated opponent reply head conditioned on afterstate.

### Step 4 — Add latent dynamics model

Predict successor embedding from `(z, action)` and train with consistency losses.

### Step 5 — Add confidence-weighted Bayesian merger

Blend model predictions with memory priors based on confidence.

---

## Minimal Practical Version for This Repository

A realistic next version for this repo is:

- keep [chess_engine/encoder.py](chess_engine/encoder.py)
- keep transformer + Hopfield retrieval in [chess_engine/model.py](chess_engine/model.py)
- split the encoder output into policy/value and memory-retrieval projections
- add a short-term current-game buffer before long-term retrieval
- extend memory schema in [chess_engine/memory.py](chess_engine/memory.py)
- update training in [chess_engine/train.py](chess_engine/train.py)
- change move selection to two-ply expected evaluation

### Proposed inference loop

1. encode current board
2. encode short-term game context
3. retrieve similar long-term memories with the enriched query
4. get top-k candidate moves
5. for each candidate:
   - either run two-ply expected evaluation
   - or run a fixed-horizon single projected line
6. compute expected score
7. pick best candidate

This gives a small planner without needing a separate full search engine.

Recommended ordering:

- first validate two-ply expected evaluation
- then add single-line fixed-horizon projection as a root-move comparison tool
- then make the projection horizon adaptive to confidence
- only later consider beam search or broader branching

---

## Why This Is Better Than Memory Alone

Memory alone only says:

- "this state resembles previous states"

But strong play needs:

- "what happens next if I choose this move and they answer well?"

So:

- memory = analogy prior
- opponent model = adversarial forecast
- value model = utility estimate
- planner = decision mechanism

All four are needed for useful non-language forward thinking.

---

## Risks

### 1. Retrieval collapse

If embeddings are not discriminative, memory becomes uniform and useless.

Mitigation:

- use a separate retrieval projection head
- track retrieval entropy explicitly during training
- reject architecture changes that improve policy loss but flatten memory attention

### 2. Wrong targets

If memory stores mismatched positions or values, planning is poisoned.

This applies separately to both memory systems:

- long-term targets must reflect reusable positional truth
- short-term targets must reflect the actual in-game trajectory

### 3. Overtrusting analogy

A similar state may differ tactically in one critical detail.

Mitigation:

- use similarity-threshold-based trust before learning confidence
- let direct model prediction dominate when retrieval is weak

### 4. Cost explosion

Top-k × top-r evaluation grows quickly.

Recommended start:

- `k = 5`
- `r = 3`

Additional risk:

### 5. Stateless drift

If the model evaluates each position in isolation, it may flip-flop between incompatible plans.

Mitigation:

- use short-term game memory to preserve trajectory consistency
- include evaluation deltas so the model can track whether a plan is working

### 6. Projection fiction

A long recursive projection can become coherent but wrong if an early predicted move is poor.

Mitigation:

- use projected lines for relative comparison of root moves, not literal forecasting
- monitor value drift across the rollout
- compare projected trajectories against actual game continuations during evaluation

Concrete projection-stability check:

- track average absolute value change per ply across the rollout
- large per-ply swings suggest unstable or fictional dynamics unless there is a clear tactical reason
- if volatility exceeds a threshold, discard or downweight that projected line
- when discarded, fall back to the simpler two-ply score for that candidate

### 7. Overconfident depth

If the confidence estimate is poorly calibrated, the system may project too far in unreliable situations.

Mitigation:

- start with fixed heuristics rather than a learned horizon controller
- cap the maximum rollout depth conservatively
- audit horizon choice against actual continuation error

Validation idea:

- take a position reached through a distinct strategic trajectory
- evaluate it once with full recent-game history
- evaluate it again with the same board but no short-term history
- if move choice or continuation scoring changes in a sensible way, short-term memory is contributing to plan continuity
- if behavior is unchanged across many such tests, the short-term path is likely being ignored

---

## Success Criteria

The new system is working if:

- memory attention becomes position-specific
- different states retrieve meaningfully different episodes
- short-term context changes retrieval in sensible ways
- the model maintains more coherent plans across consecutive moves
- predicted opponent replies are better than naive policy reuse
- two-ply expected scoring beats one-shot move selection
- fixed-horizon projected trajectories improve root-move choice beyond two-ply alone
- adaptive-depth projection beats fixed-depth projection at equal or lower compute
- ablations show memory improves move quality in familiar positions
- memory curation keeps useful patterns while preventing store pollution

Additional concrete short-term-memory test:

- present the same board with and without the recent trajectory that produced it
- confirm that positions involving committed plans (for example, pawn storms, sacrificial attacks, or long-term king-side pressure) lead to meaningfully different move preferences when history is included

---

## Bottom Line

The proposed target model is:

**a memory-assisted Bayesian-style chess planner**

Not just:

- a policy network
- a value network
- a Hopfield retrieval system

But a combined system where:

- memory provides analogical priors
- the opponent model predicts likely replies
- the planner evaluates short futures
- the value head chooses moves by expected consequence

This is the most coherent path from the current codebase toward genuine forward reasoning without language.
