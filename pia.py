#!/usr/bin/env python3
# Ansh Sapra: 501165072
# Zehra Marziya Cengiz: 
import time
from typing import Dict, List, Tuple

GAMMA = 0.95
THETA = 0.001


TERMINALS = {(0, 0), (3, 3)}

ACTIONS = ["up", "down", "right", "left"]

DELTA = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "right": (0, 1),
    "left":  (0, -1),
}

ADJACENT_ACTIONS = {
    "up":    ["left", "right"],
    "down":  ["left", "right"],
    "left":  ["up", "down"],
    "right": ["up", "down"],
}

def in_bounds(r: int, c: int, n: int = 4) -> bool:
    return 0 <= r < n and 0 <= c < n

def step_from(state: Tuple[int, int], action: str) -> Tuple[int, int]:
    r, c = state
    dr, dc = DELTA[action]
    return (r + dr, c + dc)

def next_state_distribution(
    s: Tuple[int, int], a: str, p1: float, p2: float
) -> List[Tuple[Tuple[int, int], float]]:
    
    if s in TERMINALS:
        return [(s, 1.0)]

    r, c = s
    intended = step_from(s, a)
    adj_as = ADJACENT_ACTIONS[a]
    adj1 = step_from(s, adj_as[0])
    adj2 = step_from(s, adj_as[1])

    padj = (1.0 - p1 - p2) / 2.0

    dist: Dict[Tuple[int, int], float] = {}

    def add(ns: Tuple[int, int], prob: float) -> None:
        dist[ns] = dist.get(ns, 0.0) + prob

    intended_in = in_bounds(*intended)
    if intended_in:
        add(intended, p1)
        add((r, c), p2)
    else:
        add((r, c), p1 + p2)

    add(adj1 if in_bounds(*adj1) else (r, c), padj)
    add(adj2 if in_bounds(*adj2) else (r, c), padj)

    return list(dist.items())

def print_policy(policy: Dict[Tuple[int, int], str]) -> None:
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    for r in range(4):
        row = []
        for c in range(4):
            s = (r, c)
            if s in TERMINALS:
                row.append("T")
            else:
                row.append(arrow[policy[s]])
        print(" ".join(row))

def main() -> None:
    print("Policy Iteration for generalized 4x4 Gridworld")
    print("Enter: p1 p2 rup rdown rright rleft")
    vals = input("> ").strip().split()
    if len(vals) != 6:
        raise SystemExit("Expected 6 numbers: p1 p2 rup rdown rright rleft")

    p1, p2, rup, rdown, rright, rleft = map(float, vals)
    r_of_a = {"up": rup, "down": rdown, "right": rright, "left": rleft}

    states = [(r, c) for r in range(4) for c in range(4)]

    
    pi: Dict[Tuple[int, int], Dict[str, float]] = {
        s: {a: 1.0 / 4.0 for a in ACTIONS} for s in states
    }

    
    V: Dict[Tuple[int, int], float] = {s: 0.0 for s in states}

    policy_iters = 0
    per_outer_times: List[float] = []

    while True:
        policy_iters += 1
        t0 = time.perf_counter()

        
        while True:
            delta = 0.0
            V_new = dict(V)

            for s in states:
                if s in TERMINALS:
                    V_new[s] = 0.0
                    continue

                vs = 0.0
                for a in ACTIONS:
                    pa = pi[s][a]
                    if pa == 0.0:
                        continue
                    exp = 0.0
                    for sp, prob in next_state_distribution(s, a, p1, p2):
                        exp += prob * (r_of_a[a] + GAMMA * V[sp])
                    vs += pa * exp

                V_new[s] = vs
                delta = max(delta, abs(V_new[s] - V[s]))

            V = V_new
            if delta < THETA:
                break

        
        policy_stable = True
        for s in states:
            if s in TERMINALS:
                continue

            old_best = max(pi[s], key=lambda a: pi[s][a])

            qvals: Dict[str, float] = {}
            for a in ACTIONS:
                exp = 0.0
                for sp, prob in next_state_distribution(s, a, p1, p2):
                    exp += prob * (r_of_a[a] + GAMMA * V[sp])
                qvals[a] = exp

            best_val = max(qvals.values())
            best_actions = [a for a, v in qvals.items() if abs(v - best_val) < 1e-12]

            for a in ACTIONS:
                pi[s][a] = 1.0 / len(best_actions) if a in best_actions else 0.0

            new_best = max(pi[s], key=lambda a: pi[s][a])
            if new_best != old_best:
                policy_stable = False

        t1 = time.perf_counter()
        per_outer_times.append(t1 - t0)

        if policy_stable:
            break

    final_policy: Dict[Tuple[int, int], str] = {}
    for s in states:
        if s in TERMINALS:
            final_policy[s] = "up"  
        else:
            final_policy[s] = max(pi[s], key=lambda a: pi[s][a])

    print("\n=== Results ===")
    print(f"Policy-iteration outer loops: {policy_iters}")
    print("Time per outer iteration (seconds):")
    for i, dt in enumerate(per_outer_times, start=1):
        print(f"  iter {i:3d}: {dt:.6f}")

    print("\nOptimal policy (T = terminal):")
    print_policy(final_policy)

if __name__ == "__main__":
    main()
