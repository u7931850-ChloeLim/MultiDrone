# experiment_rrt.py  (HEADLESS: no plotting, no windows)

import time, math, yaml, gc
import numpy as np
from multi_drone import MultiDrone
from rrt_planner import rrt

# -------- Headless wrapper (disable all plotting) --------
class HeadlessMultiDrone(MultiDrone):
    def _init_plot(self):
        # do not create a Plotter or any VTK objects
        self._plotter = None
        self._drone_visuals = []

    def _update_plot(self):
        # no rendering during experiments
        pass

    def visualize_paths(self, path):
        # move to final config without drawing
        self.set_configuration(path[-1])

# ---------- Stats ----------
def mean_ci_95(values):
    x = np.asarray(values, float)
    n = len(x)
    if n == 0: return np.nan, np.nan, np.nan
    mean = x.mean()
    sd = x.std(ddof=1) if n > 1 else 0.0
    half = 1.96 * sd / np.sqrt(n)
    return mean, mean - half, mean + half

def wilson_ci_95(successes, n):
    if n == 0: return np.nan, np.nan, np.nan
    z = 1.96
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return p, center - half, center + half

# ---------- Helpers ----------
def get_num_drones_from_yaml(env_path):
    with open(env_path, "r") as f:
        data = yaml.safe_load(f)
    return len(data["initial_configuration"])

def run_trial(env_path, seed, K=20000, D=2.0, timeout_s=200.0):
    np.random.seed(seed)
    N = get_num_drones_from_yaml(env_path)
    sim = HeadlessMultiDrone(num_drones=N, environment_file=env_path)

    try:
        q_start = sim.initial_configuration.astype(np.float32)
        q_goal  = sim.goal_positions.astype(np.float32)

        t0 = time.time()
        path = rrt(q_start, K, D, sim, q_goal)
        elapsed = time.time() - t0

        if elapsed > timeout_s:
            return False, float(elapsed), None

        success = (path is not None) and sim.is_goal(path[-1])
        path_len = None
        if success:
            total = 0.0
            for a, b in zip(path[:-1], path[1:]):
                total += np.linalg.norm((b - a).reshape(-1))
            path_len = float(total)

        return success, float(elapsed), path_len

    finally:
        # make sure every trial frees memory
        del sim
        gc.collect()

# ---------- Main ----------
def main():
    scenarios = [
        "environment1.yaml",
        "environment2.yaml",
        "environment3.yaml",
        "environment4.yaml",
        "environment5.yaml",
    ]
    RUNS = 50
    seeds = list(range(RUNS))
    K, D = 20000, 2.0
    TIMEOUT = 200.0

    for env in scenarios:
        results = []
        print(f"\n=== {env} | runs={RUNS}, K={K}, D={D} ===")
        for sd in seeds:
            try:
                ok, t, L = run_trial(env, sd, K=K, D=D, timeout_s=TIMEOUT)
            except Exception as e:
                print(f"  [seed {sd}] error: {e}")
                ok, t, L = False, np.nan, None
            results.append((ok, t, L))

            if (sd + 1) % 10 == 0:
                done = sd + 1
                succ = sum(1 for s,_,_ in results if s)
                print(f"  progress: {done}/{RUNS} runs | successes so far: {succ}")

        n = len(results)
        succ = sum(1 for s,_,_ in results if s)
        times = [t for s,t,_ in results if s and np.isfinite(t)]
        lengths = [L for s,_,L in results if s and L is not None]

        p, plo, phi = wilson_ci_95(succ, n)
        tmean, tlo, thi = mean_ci_95(times) if times else (np.nan, np.nan, np.nan)
        Lmean, Llo, Lhi = mean_ci_95(lengths) if lengths else (np.nan, np.nan, np.nan)

        print(f"  Success rate: {p:.2f} [{plo:.2f}, {phi:.2f}] (n={n})")
        print(f"  Time (s, success only): {tmean:.2f} [{tlo:.2f}, {thi:.2f}]")
        print(f"  Path length (success only): {Lmean:.2f} [{Llo:.2f}, {Lhi:.2f}]")

if __name__ == "__main__":
    main()
