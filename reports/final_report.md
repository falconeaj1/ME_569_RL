---
geometry: margin=0.75in
---

# Implementing and Evaluating Monte Carlo Tree Search for Connect Four

**Author:** Andrew Falcone  
**Course:** ME 569  
**Date:** June 5, 2026

## Abstract

The main problem we want to address in this paper is how well does Monte Carlo Tree Search methods work for playing and solving the game of Connect 4. The main analysis of this project is to compare how different reinforcement learning approaches using MCTS compare in different metrics such as winning, time, and compute. The main idea of MCTS is to form tree nodes and to visit states randomly at first, and then evaluate paths using a rollout approach until finding leaf node of win/tie/ or loss and back propagating back to original node, then sampling a certain number of iterations and choosing best action by using most visited state. Note that choosing state depends on both which have performed well and an exploration constant c so that both exploration and exploitation are balanced. Different rollout approaches were examined in this project including making random moves, which is quick but often does not explore the best paths, a heuristic rollout which computes immediate wins/blocks/ and a simple function for determining how good a state is, and a value function which precomputes a variety of games from the MCTS rollouts and then skips rollouts entirely and makes decisions on an immediate value function being even faster than random rollout, but not as effective as a heuristic approach for solving game. In the end a heuristic rollout was the strongest for playing against the baseline model, but with enough iterations random moves seem to slowly converge. Not enough time was spent fine tuning the value model and if more time was available, would want to pursue replicating full alpha go approach.

- Problem: evaluating Monte Carlo Tree Search for Connect Four.
- Methods: UCT MCTS, random/heuristic/value rollouts, baseline agents, fixed compute budgets.
- Main results: TODO summarize strongest findings from final plots.
- Conclusion: TODO one sentence on what mattered most: simulation budget, exploration constant, rollout policy, etc.

## 1. Introduction

Connect Four is a useful test environment for studying search and reinforcement learning for several reasons. It is a completely solved game so the state size is not as immeasurable as something like go or chess- neither of which have been solved and are orders of magnitude greater in state space. The game is also much more interesting than something like tictactoe which was used as a testbed for debugging and making MCTS algorithm generalizable. Both games share being deterministic and adversarial, but MCTS shines much for tic-tac-toe as the full width search is expensive. The project focuses on how MCTS behavior changes under compute constraints and different rollout policy choices. ADD CITATIONS for MCTS/UCT and related literature (Not sure about TD/alpha work since not directly using them but inspired by them)


Short project-goal summary:

This project implemented a reusable Connect Four environment, baseline agents, UCT MCTS, an interactive visualizer, and automated experiments for agent comparison.

## 2. Background

### 2.1 Connect Four Environment

TODO: Describe the board, players, actions, terminal states, and reward convention.
Board is array of 6x7 starting with 0s, each player is represented by +1 and -1. Game ends when one player can connect 4 pieces horizontally, vertically, or diagonally and the game is a tie if no winner and no legal moves. The actions of the game are the 7 columns, but some may become available as the game progresses due to column being full. The reward convention is that the current player always is treated as plus 1 and when the player switches, the board gets negated (double check this). The environment was made into a gymnasium wrapper to be able to immediately use reinforcement learning algorithms. 

Implementation details:

- Board size: 6 rows x 7 columns.
- Action: choose a column.
- Terminal outcomes: Red win, Yellow win, or tie.
- MCTS value convention: `+1` root player wins, `0` tie, `-1` root player loses.

### 2.2 Monte Carlo Tree Search

The Monte Carlo Tree Search algorithm is to essentially attempt to balance exploitation and exploration by making a dictionary of children nodes including their visited value and frequency count. Nodes that have not been visited much are weighted higher as well as states that have been visited and have resulted in winning states. After an action and next state is selected, then a rollout is performed to see how this state resolves. Several different rollout approaches were examined in this project including a random move policy and a simple heuristic. The rollout is played until game completion and then the end result of the game is back propagated to each early node in the tree. Upon choosing the next action, the UCT selection equation is used:

```text
score = mean_value + c * sqrt(log(parent_visits) / child_visits)
```

Where :

- `mean_value`: exploitation term.
- `c`: exploration constant.
- square-root term: exploration bonus for less-visited children.

This implementation uses UCT MCTS with random rollouts by default. UCB1 is the exploration-exploitation rule used inside UCT tree selection.

### 2.3 Baseline Agents

The following baseline agents were used to compare different algorithm approaches:

- `random`: uniformly samples legal columns.
- `bad`: deliberately weak heuristic that prefers low-scoring actions.
- `heuristic`: immediate win, immediate block, then one-ply board score.
- `mcts`: UCT MCTS with configurable simulation budget, exploration constant, and rollout policy.

Heuristic scoring:

TODO: Explain center control, open two-in-a-row windows, open three-in-a-row windows, and opponent-threat penalties.

## 3. Methods

### 3.1 Software Architecture

TODO: Briefly describe the code organization.

Important files:

- `src/connect4/core.py`: game state and rules.
- `src/connect4/mcts.py`: game-independent MCTS.
- `src/connect4/agents.py`: baseline and MCTS agents.
- `src/connect4/env.py`: Gymnasium-style environment.
- `src/connect4/pygame_app.py`: interactive visualizer.
- `src/connect4/scripts/evaluate_agents.py`: automated experiments.

### 3.2 Tic Tac Toe Testbed

TODO: Explain why Tic Tac Toe was used before Connect Four.

Include:

- Same MCTS interface as Connect Four.
- Small solved game.
- GUI shows exact minimax values.
- Used to verify MCTS data flow before scaling up.

Mention command:

```bash
tictactoe-mcts --iterations 300 --seed 0 --verbose
tictactoe-play
```

### 3.3 Experimental Protocol

TODO: Describe common settings used across experiments.

Protocol:

- Fixed random seed: TODO
- Games per condition: TODO, likely 100 or 200.
- Color swapping: MCTS plays both Red and Yellow where relevant.
- Metrics:
  - win rate
  - tie rate
  - average moves per game
  - seconds per game
  - seconds per move
  - MCTS seconds per move
  - simulations per MCTS move

Confidence interval:

```text
p_hat = wins / n
SE = sqrt(p_hat * (1 - p_hat) / n)
95% CI = p_hat +/- 1.96 * SE
```

Note: This report uses the normal approximation for win-rate confidence intervals.

### 3.4 Experiments

#### Experiment 1: Baseline Agent Matchups

Purpose:

TODO: Show that agent strength ordering is sensible.

Command:

```bash
connect4-evaluate matchups --agents random,bad,heuristic,mcts:100,mcts:500 --games TODO --seed 0 --progress-every 10
```

Expected output:

- Compact table of win rates.
- Use this as a sanity-check table, not the main result figure.

#### Experiment 2: Simulation Budget Sweep

Purpose:

TODO: Measure how MCTS performance changes as simulations per move increase.

Command:

```bash
connect4-evaluate c-sweep --iterations 10,50,100,500,1000 --exploration-weights sqrt2 --games-per-side TODO --seed 0 --csv outputs/connect4_budget_sweep.csv --plot outputs/connect4_budget_sweep.png --progress-every 10
```

Controlled variables:

- Exploration constant fixed at `sqrt(2)`.
- Rollout policy fixed to random.
- Opponent fixed to heuristic.
- MCTS plays both first and second player.

Figure placeholder:

**Figure 1. MCTS win rate vs simulations per move.**  
TODO: Insert `outputs/connect4_budget_sweep.png`.

Interpretation:

TODO: Describe whether performance improves with more simulations, where returns diminish, and whether runtime cost grows linearly.

#### Experiment 3: Exploration Constant Sweep

Purpose:

TODO: Measure sensitivity to UCT exploration constant `c`.

Command:

```bash
connect4-evaluate c-sweep --iterations 100,500 --exploration-weights 0.25,0.5,1.0,sqrt2,2.0,4.0 --games-per-side TODO --seed 0 --progress-every 10
```

Controlled variables:

- Rollout policy fixed to random.
- Opponent fixed to heuristic.
- Iteration budgets tested at 100 and 500.
- MCTS plays both first and second player.

Figure placeholder:

**Figure 2. MCTS win rate vs UCT exploration constant.**  
TODO: Insert generated c-sweep plot.

Interpretation:

TODO: Identify whether low or high exploration worked better and whether the best `c` changed with simulation budget.

#### Experiment 4: Rollout Policy Comparison

Purpose:

TODO: Compare random rollouts, heuristic-guided rollouts, and optionally value-network cutoffs.

Command without value model:

```bash
connect4-evaluate rollout-policy-sweep --policies random,heuristic --iterations 100,500 --games-per-side TODO --seed 0 --progress-every 10
```

Optional value-model workflow:

```bash
connect4-generate-value-data --games TODO --seed 0
connect4-train-value-model --epochs 20 --seed 0
connect4-evaluate rollout-policy-sweep --policies random,heuristic,value --value-model outputs/value_models/connect4_value_model.npz --value-depth 8 --iterations 100,500 --games-per-side TODO --seed 0 --progress-every 10
```

Figure placeholder:

**Figure 3. Rollout policy comparison.**  
TODO: Insert rollout-policy plot.

Interpretation:

TODO: Compare performance per simulation and per second. Note whether heuristic/value rollouts improved playing strength enough to justify their extra compute cost.

## 4. Results

### 4.1 Baseline Matchups

TODO: Insert compact table.

Table placeholder:

| Red agent | Yellow agent | Red win % | Yellow win % | Tie % | Avg moves |
|---|---:|---:|---:|---:|---:|
| TODO | TODO | TODO | TODO | TODO | TODO |

Interpretation:

TODO: Summarize the agent ordering. Mention whether stronger MCTS beat weaker MCTS and heuristic baselines.

### 4.2 Simulation Budget

TODO: Insert Figure 1.

Key observations:

- TODO
- TODO
- TODO

### 4.3 Exploration Constant

TODO: Insert Figure 2.

Key observations:

- TODO
- TODO
- TODO

### 4.4 Rollout Policy

TODO: Insert Figure 3.

Key observations:

- TODO
- TODO
- TODO

### 4.5 Runtime Cost

TODO: Summarize compute-cost metrics.

Suggested table:

| Condition | Simulations/move | MCTS seconds/move | Win rate |
|---|---:|---:|---:|
| TODO | TODO | TODO | TODO |

Interpretation:

TODO: Explain tradeoff between strength and runtime.

## 5. Discussion

TODO: Discuss what worked, what did not, and why.

Prompts:

- Did increasing simulations reliably improve performance?
- Was `sqrt(2)` a reasonable exploration constant?
- Did heuristic-guided rollouts help or hurt?
- Did value-network cutoff help, or was it too noisy/preliminary?
- How did first-player advantage affect interpretation?
- What are the limitations of using win rate against heuristic as the main metric?

Limitations:

- Connect Four is not solved by this implementation.
- MCTS searches are stochastic.
- Experiments use finite samples.
- Neural value function, if included, is trained from generated self-play outcomes, not perfect-play labels.
- Runtime depends on hardware.

## 6. Summary And Future Work

TODO: Recap the main findings in 1-3 paragraphs.

Future work ideas:

- Reuse MCTS trees between moves.
- Add transposition tables.
- Train stronger value/policy networks.
- Compare against tabular Q-learning on reduced board sizes.
- Evaluate tactical benchmark positions.
- Increase experiment repetitions for tighter confidence intervals.

## Code And Data Availability

Code repository:

TODO: Add GitHub link.

Generated outputs:

TODO: List output CSV/PNG files used in report.

Example:

- `outputs/connect4_budget_sweep.csv`
- `outputs/connect4_budget_sweep.png`
- `outputs/connect4_c_sweep.csv`
- `outputs/connect4_rollout_policy_sweep.csv`

## Contributions

Andrew Falcone:

- Project scope and implementation.
- Connect Four environment and agents.
- MCTS implementation.
- Evaluation scripts.
- GUI visualizers.
- Experiments and report writing.

## Bibliography

TODO: Add references from literature review.

Suggested citation topics:

- UCT / MCTS original work.
- Temporal-difference learning.
- AlphaGo / AlphaZero for MCTS with value/policy networks.
- Any course texts or lecture notes used for RL background.
