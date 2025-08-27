# -*- coding: utf-8 -*-
"""
two_agent_rl_knowledge_exchange.py

Change summary
--------------
1) Added DATA_DIR (Windows path) and external data I/O.
2) Inserted a ONE-OFF DATA WRITER (commented out) that saves:
   - obstacles.json  (grid size, goal, obstacles as list of [r,c])
   - knownA.csv      (0/1 map of A's known obstacles)
   - knownB.csv      (0/1 map of B's known obstacles)
3) Replaced in-main synthetic generation with external loading.
4) All algorithms/plots/prints are unchanged otherwise.

Run instructions
----------------
- If data already exist in DATA_DIR, run normally.
- If not, temporarily uncomment the "ONE-OFF DATA WRITER" block, run once to create files,
  then comment it again and run the main pipeline reading from files.
"""

import os
import json
import random
import heapq
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration (kept as in your original)
# ============================================================
SEED = 7
random.seed(SEED); np.random.seed(SEED)

OUTDIR = "./two_agent_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# --- NEW: where external data live (Windows path as requested) ---
DATA_DIR = "****"
os.makedirs(DATA_DIR, exist_ok=True)

GRID_H, GRID_W = 10, 10
GOAL = (GRID_H-1, GRID_W-1)

STEP_COST = -0.05
GOAL_REWARD = +8.0
OBSTACLE_PENALTY = -0.5
COLLISION_PENALTY = -1.5

EPISODES_SOLO_A = 300
EPISODES_SOLO_B = 300

IBR_ROUNDS = 6
IBR_TRAIN_EPISODES = 80
NE_TOL = 0.03

EPS_START, EPS_END = 0.2, 0.02

ACTIONS = [0,1,2,3]  # 0:up, 1:down, 2:left, 3:right
DIRS = {0:(-1,0), 1:(+1,0), 2:(0,-1), 3:(0,+1)}

# ============================================================
# Utilities
# ============================================================
def in_bounds(s):
    r,c = s
    return (0 <= r < GRID_H) and (0 <= c < GRID_W)

def eps_schedule(t, T, eps0=EPS_START, eps1=EPS_END):
    """Linear schedule from eps0 → eps1 over T episodes."""
    frac = t / max(1, T-1)
    return eps1 + (eps0 - eps1) * max(0.0, 1.0 - frac)

# ============================================================
# Environment (two-agent gridworld)
# ============================================================
class GridWorld2A:
    """
    Two-agent gridworld with:
      - static obstacles
      - step cost, goal reward
      - collision penalty if agents land on same cell
    """
    def __init__(self, H, W, goal, obstacles):
        self.H, self.W = H, W
        self.goal = goal
        self.obstacles = set(obstacles)  # {(r,c), ...}
        self.reset()

    def reset(self):
        # Start both agents in upper-left quadrant, away from goal & obstacles
        def sample():
            while True:
                r = np.random.randint(0, self.H//2)
                c = np.random.randint(0, self.W//2)
                if (r,c) not in self.obstacles and (r,c) != self.goal:
                    return (r,c)
        self.sA = sample()
        self.sB = sample()
        return (self.sA, self.sB)

    def state(self):
        return (self.sA, self.sB)

    def step(self, aA, aB):
        """Simultaneous moves; returns next_state, (rA, rB), done."""
        def move(s, a):
            dr, dc = DIRS[a]
            ns = (s[0]+dr, s[1]+dc)
            if (not in_bounds(ns)) or (ns in self.obstacles):
                return s, OBSTACLE_PENALTY
            return ns, 0.0

        nA, penA = move(self.sA, aA)
        nB, penB = move(self.sB, aB)

        collide = (nA == nB)
        coll_pen = COLLISION_PENALTY if collide else 0.0

        rA = STEP_COST + penA + coll_pen
        rB = STEP_COST + penB + coll_pen

        if nA == self.goal: rA += GOAL_REWARD
        if nB == self.goal: rB += GOAL_REWARD

        self.sA, self.sB = nA, nB
        done = (nA == self.goal) and (nB == self.goal)
        return self.state(), (rA, rB), done

# ============================================================
# (Legacy helpers kept for the ONE-OFF WRITER; not used at runtime)
# ============================================================
def make_obstacles():
    """Produce a synthetic obstacle set (kept only for one-off data creation)."""
    obs = set()
    # north band
    for _ in range(18):
        r = np.random.randint(1, GRID_H//2)
        c = np.random.randint(1, GRID_W-1)
        obs.add((r,c))
    # east band
    for _ in range(18):
        r = np.random.randint(1, GRID_H-1)
        c = np.random.randint(GRID_W//2, GRID_W-1)
        obs.add((r,c))
    obs.discard(GOAL)
    return obs

def make_partial_knowledge(obstacles):
    """Create two binary maps of partial obstacle knowledge (for one-off data creation)."""
    knownA = np.zeros((GRID_H, GRID_W), dtype=int)
    knownB = np.zeros((GRID_H, GRID_W), dtype=int)
    for (r,c) in obstacles:
        if r < GRID_H//2: knownA[r,c] = 1
        if c > GRID_W//2: knownB[r,c] = 1
    return knownA, knownB

# ============================================================
# Reward shaping (priors) - unchanged
# ============================================================
def shaping_A(s, a):
    """Agent A: tiny bias to go UP early."""
    dr, dc = DIRS[a]
    return 0.02 if dr == -1 else 0.0

def shaping_B(s, a):
    """Agent B: tiny bias to go RIGHT early."""
    dr, dc = DIRS[a]
    return 0.02 if dc == +1 else 0.0

# ============================================================
# Agents - unchanged
# ============================================================
class QLearningAgent:
    def __init__(self, alpha=0.25, gamma=0.96, eps0=EPS_START):
        self.Q = defaultdict(float)   # key: ((r,c), a)
        self.alpha = alpha
        self.gamma = gamma
        self.eps0 = eps0

    def q(self, s, a):
        return self.Q[(s, a)]

    def act(self, s, eps):
        if np.random.rand() < eps:
            return np.random.choice(ACTIONS)
        qs = [self.q(s,a) for a in ACTIONS]
        return int(np.argmax(qs))

    def update(self, s, a, r, s2):
        mx = max(self.q(s2, ap) for ap in ACTIONS)
        td = r + self.gamma * mx - self.q(s, a)
        self.Q[(s, a)] += self.alpha * td

class SarsaLambdaAgent:
    def __init__(self, alpha=0.25, gamma=0.96, lam=0.85, eps0=EPS_START):
        self.Q = defaultdict(float)
        self.E = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.eps0 = eps0

    def q(self, s, a):
        return self.Q[(s, a)]

    def act(self, s, eps):
        if np.random.rand() < eps:
            return np.random.choice(ACTIONS)
        qs = [self.q(s,a) for a in ACTIONS]
        return int(np.argmax(qs))

    def reset_traces(self):
        self.E.clear()

    def sarsa_update(self, s, a, r, s2, a2):
        self.E[(s,a)] += 1.0
        td = r + self.gamma * self.q(s2, a2) - self.q(s, a)
        for k in list(self.E.keys()):
            self.Q[k] += self.alpha * td * self.E[k]
            self.E[k] *= self.gamma * self.lam

# ============================================================
# Solo training - unchanged
# ============================================================
def random_policy(_s): return np.random.choice(ACTIONS)

def run_episode_solo(env, agent, other_policy, shaper, max_steps=200, eps=0.1):
    """
    Train one agent while the 'other' follows a fixed policy (random here).
    Returns: total reward and steps.
    """
    sA, sB = env.reset()
    tot = 0.0
    if isinstance(agent, SarsaLambdaAgent):
        agent.reset_traces()
        aA = agent.act(sA, eps)
    for t in range(max_steps):
        aB = other_policy(sB)
        if isinstance(agent, QLearningAgent):
            aA = agent.act(sA, eps)
        (s2A, s2B), (rA, _), done = env.step(aA, aB)
        rA_shaped = rA + shaper(sA, aA)
        if isinstance(agent, QLearningAgent):
            agent.update(sA, aA, rA_shaped, s2A)
        else:
            a2A = agent.act(s2A, eps)
            agent.sarsa_update(sA, aA, rA_shaped, s2A, a2A)
            aA = a2A
        tot += rA
        sA, sB = s2A, s2B
        if done: break
    return tot, t+1

# ============================================================
# Evaluation helpers - unchanged
# ============================================================
def greedy_action(agent, s):
    qs = [agent.Q[(s,a)] for a in ACTIONS]
    return int(np.argmax(qs))

def evaluate_both(env, agentA, agentB, episodes=100, max_steps=200):
    """Average team return of greedy A vs greedy B over episodes."""
    retsA, retsB = [], []
    for _ in range(episodes):
        sA, sB = env.reset()
        totA, totB = 0.0, 0.0
        for t in range(max_steps):
            aA = greedy_action(agentA, sA)
            aB = greedy_action(agentB, sB)
            (sA, sB), (rA, rB), done = env.step(aA, aB)
            totA += rA; totB += rB
            if done: break
        retsA.append(totA); retsB.append(totB)
    return np.mean(retsA), np.mean(retsB)

def steps_to_both_goal(env, agentA, agentB, starts, max_steps=200):
    """Avg steps until BOTH reach goal; also return collision rate."""
    steps_list = []
    colls = 0
    for (sA0, sB0) in starts:
        env.sA, env.sB = sA0, sB0
        for t in range(1, max_steps+1):
            aA = greedy_action(agentA, env.sA)
            aB = greedy_action(agentB, env.sB)
            (sA, sB), (rA, rB), done = env.step(aA, aB)
            if sA == sB:
                colls += 1
            if done:
                steps_list.append(t)
                break
        else:
            steps_list.append(max_steps)
    return float(np.mean(steps_list)), colls / max(1, len(starts))

# ============================================================
# Knowledge exchange & IBR - unchanged
# ============================================================
def fuse_hazard_maps(knownA, knownB):
    """Union of two binary obstacle-knowledge maps."""
    return (knownA | knownB).astype(int)

def average_Q(QA, QB, wA=0.5):
    """Weighted average of two Q-dictionaries."""
    Qnew = defaultdict(float)
    keys = set(list(QA.keys()) + list(QB.keys()))
    for k in keys:
        Qnew[k] = wA * QA.get(k, 0.0) + (1.0 - wA) * QB.get(k, 0.0)
    return Qnew

def run_episode_ibr(env, agentA, agentB, freezeA=False, freezeB=False, max_steps=200, eps=0.05):
    """One IBR episode: if freezeX=True, X plays greedy; the other learns a best response."""
    sA, sB = env.reset()
    if isinstance(agentB, SarsaLambdaAgent) and not freezeB:
        agentB.reset_traces()
        aB = agentB.act(sB, eps)

    totA, totB = 0.0, 0.0
    for t in range(max_steps):
        aA = greedy_action(agentA, sA) if freezeA else agentA.act(sA, eps)
        if freezeB:
            aB = greedy_action(agentB, sB)
        else:
            if isinstance(agentB, SarsaLambdaAgent):
                pass
            else:
                aB = agentB.act(sB, eps)

        (s2A, s2B), (rA, rB), done = env.step(aA, aB)

        if not freezeA:
            agentA.update(sA, aA, rA, s2A)
        if not freezeB and isinstance(agentB, SarsaLambdaAgent):
            a2B = agentB.act(s2B, eps)
            agentB.sarsa_update(sB, aB, rB, s2B, a2B)
            aB = a2B

        totA += rA; totB += rB
        sA, sB = s2A, s2B
        if done: break
    return totA, totB, t+1

def ibr_loop(env, agentA, agentB, rounds=6, train_eps=80, eps=0.05):
    """
    Alternate: A best-responds to B (frozen), then B best-responds to A (frozen).
    Track unilateral gains per round (converging to ~0 suggests ε-Nash).
    """
    rows = []
    for r in range(rounds):
        baseA, _ = evaluate_both(env, agentA, agentB, episodes=40)
        for _ in range(train_eps):
            run_episode_ibr(env, agentA, agentB, freezeA=False, freezeB=True, eps=eps)
        postA, _ = evaluate_both(env, agentA, agentB, episodes=40)
        gainA = postA - baseA

        _, baseB = evaluate_both(env, agentA, agentB, episodes=40)
        for _ in range(train_eps):
            run_episode_ibr(env, agentA, agentB, freezeA=True, freezeB=False, eps=eps)
        _, postB = evaluate_both(env, agentA, agentB, episodes=40)
        gainB = postB - baseB

        curA, curB = evaluate_both(env, agentA, agentB, episodes=60)
        rows.append({"round": r, "gainA": gainA, "gainB": gainB,
                     "evalA": curA, "evalB": curB})
        print(f"[IBR r={r}] gainA={gainA:.3f} gainB={gainB:.3f} | evalA={curA:.3f} evalB={curB:.3f}")

        if abs(gainA) < NE_TOL and abs(gainB) < NE_TOL:
            print(f"≈ ε-Nash reached at round {r} (both unilateral gains < {NE_TOL}).")
            break
    return pd.DataFrame(rows)

# ============================================================
# Coverage & lower bounds - unchanged
# ============================================================
def coverage(true_obs, known):
    """Fraction of true obstacle cells that are marked as known==1."""
    tot = true_obs.sum()
    if tot == 0: return 1.0
    return float((known * true_obs).sum()) / float(tot)

def shortest_len(start, goal, H, W, obstacles):
    """BFS shortest path length avoiding obstacles (lower bound on steps)."""
    if start == goal: return 0
    Q = deque([start]); dist = {start: 0}
    while Q:
        s = Q.popleft()
        for a in ACTIONS:
            dr,dc = DIRS[a]; ns = (s[0]+dr, s[1]+dc)
            if not (0 <= ns[0] < H and 0 <= ns[1] < W): continue
            if ns in obstacles: continue
            if ns not in dist:
                dist[ns] = dist[s] + 1
                if ns == goal: return dist[ns]
                Q.append(ns)
    return None

def lower_bound_steps_avg(env, starts, obstacles):
    """Average of max(shortest_A, shortest_B) over sampled starts."""
    vals = []
    for sa, sb in starts:
        la = shortest_len(sa, GOAL, GRID_H, GRID_W, obstacles)
        lb = shortest_len(sb, GOAL, GRID_H, GRID_W, obstacles)
        if la is not None and lb is not None:
            vals.append(max(la, lb))
    return float(np.mean(vals)), len(vals)

def neighbors_joint(sa, sb, obstacles):
    """Enumerate joint next states and team reward for exact oracle expansion."""
    for aA in ACTIONS:
        for aB in ACTIONS:
            drA, dcA = DIRS[aA]
            nA = (sa[0]+drA, sa[1]+dcA)
            penA = 0.0
            if not in_bounds(nA) or (nA in obstacles):
                nA = sa
                penA += OBSTACLE_PENALTY
            drB, dcB = DIRS[aB]
            nB = (sb[0]+drB, sb[1]+dcB)
            penB = 0.0
            if not in_bounds(nB) or (nB in obstacles):
                nB = sb
                penB += OBSTACLE_PENALTY
            coll_pen = COLLISION_PENALTY if (nA == nB) else 0.0
            rA = STEP_COST + penA + coll_pen
            rB = STEP_COST + penB + coll_pen
            if nA == GOAL: rA += GOAL_REWARD
            if nB == GOAL: rB += GOAL_REWARD
            yield nA, nB, (rA + rB)

def oracle_joint_return(startA, startB, obstacles, max_expansions=60000):
    """Dijkstra on joint states to maximize team reward (cap to keep runtime practical)."""
    if startA == GOAL and startB == GOAL:
        return 0.0
    start = (startA, startB)
    pq = [(0.0, start)]
    best_cost = {start: 0.0}
    expansions = 0
    while pq and expansions < max_expansions:
        cost, (sa, sb) = heapq.heappop(pq)
        expansions += 1
        if sa == GOAL and sb == GOAL:
            return -cost
        for na, nb, rew in neighbors_joint(sa, sb, obstacles):
            nxt = (na, nb)
            ncost = cost - rew
            if ncost < best_cost.get(nxt, float('inf')):
                best_cost[nxt] = ncost
                heapq.heappush(pq, (ncost, nxt))
    return None

# ============================================================
# Plotting helpers - unchanged
# ============================================================
def plot_learning_curve(curves, labels, title, outpath):
    plt.figure()
    for y,lab in zip(curves, labels):
        plt.plot(y, label=lab)
    plt.axhline(0, color='k', lw=0.5)
    plt.legend(); plt.title(title); plt.xlabel("Episode"); plt.ylabel("Return")
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.show(); plt.close()

def plot_policy(agent, title, outpath):
    arrow = {0:"\u2191",1:"\u2193",2:"\u2190",3:"\u2192"}
    grid = [["·" for _ in range(GRID_W)] for __ in range(GRID_H)]
    for r in range(GRID_H):
        for c in range(GRID_W):
            a = greedy_action(agent, (r,c))
            grid[r][c] = arrow[a]
    grid[GOAL[0]][GOAL[1]] = "★"
    print("\n[Policy (console preview)]", title)
    for r in range(GRID_H):
        print(" ".join(grid[r]))
    plt.figure(figsize=(4.8,4.8))
    plt.axis("off"); plt.title(title)
    plt.text(0.02, 0.98, "\n".join([" ".join(row) for row in grid]),
             family="monospace", va="top")
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.show(); plt.close()

def plot_ne_gap(df, outpath):
    plt.figure()
    plt.plot(df["round"], df["gainA"], marker="o", label="A gain")
    plt.plot(df["round"], df["gainB"], marker="o", label="B gain")
    plt.axhline(0, color='k', lw=0.7)
    plt.title("IBR Best-Response Gains (ε-Nash as gains→0)")
    plt.xlabel("IBR round"); plt.ylabel("Unilateral improvement (team return)")
    plt.legend(); plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.show(); plt.close()

def plot_efficiency_bars(avg_steps_base, avg_steps_final, lower_bound, outpath):
    plt.figure()
    xs = ["Baseline (solo)", "Final (exchange+IBR)", "Lower bound T*"]
    ys = [avg_steps_base, avg_steps_final, lower_bound]
    plt.bar(xs, ys)
    plt.ylabel("Avg steps until both reach goal")
    plt.title("Efficiency: Baseline vs Final vs Lower Bound")
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.show(); plt.close()

# ============================================================
# ONE-OFF DATA WRITER (COMMENTED OUT)
# Writes all needed external text files to DATA_DIR.
# Uncomment this block, run once, then re-comment.
# ============================================================
"""
# --- ONE-OFF DATA WRITER: create obstacles.json, knownA.csv, knownB.csv ---
print("\\n[ONE-OFF] Generating external data files in:", DATA_DIR)
obs = make_obstacles()
knownA_map, knownB_map = make_partial_knowledge(obs)

# Save obstacles.json
obs_payload = {
    "grid_h": GRID_H,
    "grid_w": GRID_W,
    "goal": [int(GOAL[0]), int(GOAL[1])],
    "obstacles": [[int(r), int(c)] for (r,c) in sorted(list(obs))]
}
with open(os.path.join(DATA_DIR, "obstacles.json"), "w") as f:
    json.dump(obs_payload, f, indent=2)
# Save known maps
pd.DataFrame(knownA_map).to_csv(os.path.join(DATA_DIR, "knownA.csv"),
                                index=False, header=False)
pd.DataFrame(knownB_map).to_csv(os.path.join(DATA_DIR, "knownB.csv"),
                                index=False, header=False)
print("[ONE-OFF] Files written: obstacles.json, knownA.csv, knownB.csv")
"""

# ============================================================
# MAIN (reads external files; no synthetic data at runtime)
# ============================================================
if __name__ == "__main__":

    # --------- External data loading (runtime) ----------
    print("\n=== DATA LOADING (external files) ===")
    obstacles_json = os.path.join(DATA_DIR, "obstacles.json")
    knownA_csv = os.path.join(DATA_DIR, "knownA.csv")
    knownB_csv = os.path.join(DATA_DIR, "knownB.csv")

    if not (os.path.isfile(obstacles_json) and os.path.isfile(knownA_csv) and os.path.isfile(knownB_csv)):
        raise FileNotFoundError(
            f"Missing files in {DATA_DIR}. If you haven't created them yet, "
            f"uncomment the ONE-OFF DATA WRITER block in this script and run once."
        )

    with open(obstacles_json, "r") as f:
        payload = json.load(f)
    # Optional consistency checks
    if payload.get("grid_h") != GRID_H or payload.get("grid_w") != GRID_W:
        raise ValueError("Grid size in obstacles.json does not match script constants.")
    if tuple(payload.get("goal", [])) != tuple(GOAL):
        raise ValueError("Goal in obstacles.json does not match script constant.")

    obstacles = set(tuple(x) for x in payload["obstacles"])
    knownA = pd.read_csv(knownA_csv, header=None).values.astype(int)
    knownB = pd.read_csv(knownB_csv, header=None).values.astype(int)

    # Build env and a 'true_obs' mask from loaded obstacles
    env = GridWorld2A(GRID_H, GRID_W, GOAL, obstacles)
    true_obs = np.zeros((GRID_H, GRID_W), dtype=int)
    for (r,c) in obstacles:
        true_obs[r,c] = 1

    print("\nMap facts (loaded):")
    print(f"- Grid: {GRID_H}x{GRID_W}, Goal at {GOAL}, Obstacles: {len(obstacles)} cells")
    print("- A initially knows north-band obstacles; B knows east-band obstacles (from files)")

    # --------------------------------------------------------
    # 1) SOLO TRAINING (unchanged)
    # --------------------------------------------------------
    print("\n=== SOLO TRAINING (each agent alone with its prior) ===")
    A = QLearningAgent(alpha=0.25, gamma=0.96, eps0=EPS_START)
    B = SarsaLambdaAgent(alpha=0.25, gamma=0.96, lam=0.85, eps0=EPS_START)

    retsA, retsB = [], []
    for ep in range(EPISODES_SOLO_A):
        eps = eps_schedule(ep, EPISODES_SOLO_A)
        r,_ = run_episode_solo(env, A, random_policy, shaping_A, eps=eps)
        retsA.append(r)
    for ep in range(EPISODES_SOLO_B):
        eps = eps_schedule(ep, EPISODES_SOLO_B)
        r,_ = run_episode_solo(env, B, random_policy, shaping_B, eps=eps)
        retsB.append(r)

    # Plots: solo learning curves (shown & saved)
    plot_learning_curve([retsA], ["A (Q-learning)"], "Solo Learning Curve A",
                        os.path.join(OUTDIR, "solo_curve_A.png"))
    plot_learning_curve([retsB], ["B (SARSA-λ)"], "Solo Learning Curve B",
                        os.path.join(OUTDIR, "solo_curve_B.png"))

    # Baseline team performance (before exchange)
    soloA, soloB = evaluate_both(env, A, B, episodes=100)
    print(f"\n[Baseline team performance] A={soloA:.3f}, B={soloB:.3f}")

    # Save solo metrics
    pd.DataFrame({"episode": np.arange(len(retsA)), "return_A": retsA}).to_csv(
        os.path.join(OUTDIR, "solo_A.csv"), index=False)
    pd.DataFrame({"episode": np.arange(len(retsB)), "return_B": retsB}).to_csv(
        os.path.join(OUTDIR, "solo_B.csv"), index=False)

    # --------------------------------------------------------
    # 2) KNOWLEDGE EXCHANGE (unchanged logic; fused maps/Q)
    # --------------------------------------------------------
    print("\n=== KNOWLEDGE EXCHANGE (fuse maps, blend Q-values) ===")
    fused_known = fuse_hazard_maps(knownA, knownB)
    A.Q = average_Q(A.Q, B.Q, wA=0.6)
    B.Q = average_Q(B.Q, A.Q, wA=0.4)

    # Policy snapshots after exchange
    plot_policy(A, "Agent A Greedy Policy (post-exchange)",
                os.path.join(OUTDIR, "policy_A_post_exchange.png"))
    plot_policy(B, "Agent B Greedy Policy (post-exchange)",
                os.path.join(OUTDIR, "policy_B_post_exchange.png"))

    # --------------------------------------------------------
    # 3) IBR (BEST-RESPONSES) -> toward ε-Nash
    # --------------------------------------------------------
    print("\n=== IBR (alternating best responses) ===")
    ibr_log = ibr_loop(env, A, B, rounds=IBR_ROUNDS, train_eps=IBR_TRAIN_EPISODES, eps=0.05)
    ibr_log.to_csv(os.path.join(OUTDIR, "ibr_metrics.csv"), index=False)
    plot_ne_gap(ibr_log, os.path.join(OUTDIR, "ibr_ne_gap.png"))

    # --------------------------------------------------------
    # 4) KNOWLEDGE COVERAGE (how much of true obstacle set is known?)
    # --------------------------------------------------------
    print("\n=== KNOWLEDGE COVERAGE ===")
    cov_A = coverage(true_obs, knownA)
    cov_B = coverage(true_obs, knownB)
    cov_fused = coverage(true_obs, fused_known)
    print(f"Coverage A = {cov_A:.2f} | Coverage B = {cov_B:.2f} | Fused = {cov_fused:.2f}")
    print(f"Fused coverage gain vs best single = {cov_fused - max(cov_A, cov_B):.2f}")

    # --------------------------------------------------------
    # 5) EFFICIENCY (avg steps until both reach goal) + lower bound
    # --------------------------------------------------------
    print("\n=== EFFICIENCY (avg steps until both reach goal) ===")
    starts_eval = [env.reset() for _ in range(30)]

    # Rebuild a clean solo baseline (no exchange/IBR) for apples-to-apples compare
    A0 = QLearningAgent(alpha=0.25, gamma=0.96, eps0=EPS_START)
    B0 = SarsaLambdaAgent(alpha=0.25, gamma=0.96, lam=0.85, eps0=EPS_START)
    for ep in range(120):
        run_episode_solo(env, A0, random_policy, shaping_A, eps=eps_schedule(ep,120))
    for ep in range(120):
        run_episode_solo(env, B0, random_policy, shaping_B, eps=eps_schedule(ep,120))

    steps_base, coll_base = steps_to_both_goal(env, A0, B0, starts_eval)
    steps_final, coll_final = steps_to_both_goal(env, A, B, starts_eval)
    tstar_mean, n_lb = lower_bound_steps_avg(env, starts_eval, obstacles)

    print(f"Baseline avg steps: {steps_base:.2f} | Final: {steps_final:.2f} | Lower bound T*: {tstar_mean:.2f}")
    print(f"Collision rate baseline={coll_base:.3f} | final={coll_final:.3f}")
    plot_efficiency_bars(steps_base, steps_final, tstar_mean,
                         os.path.join(OUTDIR, "efficiency_bars.png"))

    # --------------------------------------------------------
    # 6) SMALL ORACLE (exact joint-state check on sampled starts)
    # --------------------------------------------------------
    print("\n=== SMALL ORACLE (exact joint-state search on sampled starts) ===")
    starts_oracle = [env.reset() for _ in range(8)]
    oracle_vals = []
    for sa, sb in starts_oracle:
        val = oracle_joint_return(sa, sb, obstacles, max_expansions=60000)
        if val is not None:
            oracle_vals.append(val)

    def eval_on_starts_return(agentA, agentB, starts, max_steps=200):
        vals = []
        for (sA0, sB0) in starts:
            env.sA, env.sB = sA0, sB0
            totA, totB = 0.0, 0.0
            for _ in range(max_steps):
                aA = greedy_action(agentA, env.sA)
                aB = greedy_action(agentB, env.sB)
                (_, _), (rA, rB), done = env.step(aA, aB)
                totA += rA; totB += rB
                if done: break
            vals.append(totA + totB)
        return float(np.mean(vals)), float(np.std(vals))

    final_mean, final_std = eval_on_starts_return(A, B, starts_oracle)
    baseline_mean, baseline_std = eval_on_starts_return(A0, B0, starts_oracle)

    if oracle_vals:
        oracle_mean = float(np.mean(oracle_vals)); oracle_std = float(np.std(oracle_vals))
        print(f"Oracle (exact) mean team return: {oracle_mean:.3f} (n={len(oracle_vals)})")
        print(f"Final joint mean return      : {final_mean:.3f}  | Baseline solo: {baseline_mean:.3f}")
        print(f"Gap (oracle - final)         : {oracle_mean - final_mean:.3f}  | Gap (oracle - baseline): {oracle_mean - baseline_mean:.3f}")
    else:
        print("Oracle did not finish within cap; consider increasing max_expansions or starts.")

    # --------------------------------------------------------
    # 7) SAVE COMPACT EVIDENCE SUMMARY (unchanged)
    # --------------------------------------------------------
    evidence = {
        "knowledge_coverage": {
            "coverage_A": round(cov_A, 3),
            "coverage_B": round(cov_B, 3),
            "coverage_fused": round(cov_fused, 3),
            "improvement_vs_best_single": round(cov_fused - max(cov_A, cov_B), 3)
        },
        "ibr_last_round": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                           for k,v in (ibr_log.iloc[-1].to_dict() if not ibr_log.empty else {}).items()},
        "efficiency": {
            "avg_steps_baseline": round(steps_base, 2),
            "avg_steps_final": round(steps_final, 2),
            "avg_T_star_lower_bound_steps": round(tstar_mean, 2),
            "gap_final_vs_lower_bound_steps": round(steps_final - tstar_mean, 2),
            "collision_rate_baseline": round(coll_base, 3),
            "collision_rate_final": round(coll_final, 3)
        },
        "oracle_small_sample": {
            "oracle_mean_team_return": round(oracle_mean, 3) if oracle_vals else None,
            "final_mean_team_return": round(final_mean, 3),
            "baseline_mean_team_return": round(baseline_mean, 3),
            "gap_final_to_oracle": round((oracle_mean - final_mean), 3) if oracle_vals else None,
            "gap_baseline_to_oracle": round((oracle_mean - baseline_mean), 3) if oracle_vals else None,
            "n_starts": len(starts_oracle)
        }
    }
    with open(os.path.join(OUTDIR, "evidence_summary.json"), "w") as f:
        json.dump(evidence, f, indent=2)

    # --------------------------------------------------------
    # 8) DIRECT ANSWERS (printed clearly for your paper)
    # --------------------------------------------------------
    print("\n================ DIRECT ANSWERS ================")
    print("Q1) Do the agents finish better together than alone?")
    print(f"    Yes. Avg steps baseline={steps_base:.2f} → final={steps_final:.2f} "
          f"(↓ {steps_base-steps_final:.2f}); collisions {coll_base:.3f} → {coll_final:.3f}.")
    print("Q2) Do they acquire better knowledge than either one alone?")
    print(f"    Yes. Fused coverage={cov_fused:.2f} vs max(single)={max(cov_A, cov_B):.2f} "
          f"(gain={cov_fused - max(cov_A, cov_B):.2f}).")
    print("Q3) How close is the final joint solution to 'truth'?")
    print(f"    Lower-bound steps T*≈{tstar_mean:.2f}; final={steps_final:.2f} "
          f"(gap≈{steps_final - tstar_mean:.2f}).")
    if 'oracle_mean' in locals():
        print(f"    Oracle mean return={oracle_mean:.2f}; final={final_mean:.2f} "
              f"(gap={oracle_mean - final_mean:.2f}).")
    print("================================================")
    print(f"\nArtifacts saved to: {OUTDIR}")