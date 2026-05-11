# 🤖 Gridworld Reinforcement Learning – Value & Policy Iteration

Implementation of Value Iteration and Policy Iteration on a stochastic 4x4 Gridworld MDP, with configurable transition probabilities, per-action rewards, and full convergence analysis across deterministic and stochastic environments.

---

## 📌 Overview

This project solves a generalized 4x4 Gridworld using two classic dynamic programming algorithms from reinforcement learning. The agent navigates a grid with stochastic actions toward two terminal states, and both algorithms converge to the same optimal policy — demonstrating the consistency of MDP planning methods.

---

## 🗺️ Environment

- **Grid:** 4x4 with terminal states at (0,0) and (3,3)
- **Actions:** `up`, `down`, `left`, `right`
- **Transition Model:**
  - With probability `p1` → move to intended state
  - With probability `p2` → stay in place
  - With probability `(1 - p1 - p2) / 2` → drift to either adjacent state
  - Off-grid moves: stay in place with probability `p1 + p2`
- **Discount factor:** γ = 0.95
- **Convergence threshold:** θ = 0.001

---

## ⚙️ Algorithms

**`via.py` — Value Iteration**
- Initializes V(s) = 0 for all states
- Repeatedly updates each state's value using the Bellman optimality equation
- Extracts greedy policy after convergence

**`pia.py` — Policy Iteration**
- Starts with equiprobable random policy
- Alternates between full policy evaluation and greedy policy improvement
- Converges when policy is stable across iterations

---

## 📊 Results Summary

| Case | p1 | p2 | VI Iterations | PI Outer Loops |
|---|---|---|---|---|
| Deterministic | 1.0 | 0.0 | 4 | 3 |
| Mild Noise | 0.8 | 0.1 | 11 | 2 |
| High Noise | 0.5 | 0.3 | 21 | 2 |

**Key observations:**
- As stochasticity increases, Value Iteration requires significantly more iterations
- Policy Iteration converges in fewer outer loops but each loop is more expensive
- Both algorithms produce the same optimal policy across all cases

---

## ▶️ How to Run

**Requirements**
```bash
python3
```
No external libraries needed — uses only Python standard library.

**Value Iteration**
```bash
python3 via.py
```

**Policy Iteration**
```bash
python3 pia.py
```

**Input format (when prompted)**
```
> p1 p2 rup rdown rright rleft
```

**Example — Deterministic case:**
```
> 1.0 0.0 -1 -1 -1 -1
```

**Example — Stochastic case:**
```
> 0.8 0.1 -1 -1 -1 -1
```

---

## 📁 Files

| File | Description |
|---|---|
| `via.py` | Value Iteration implementation |
| `pia.py` | Policy Iteration implementation |
| `GridWorld_RL_Report.pdf` | Full results report with convergence analysis |

---

## 🛠️ Tech Stack

- Python 3
- Reinforcement Learning (MDP Planning)
- Dynamic Programming
- Type hints (`typing` module)
