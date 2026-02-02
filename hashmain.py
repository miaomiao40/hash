import hashlib
import os
import time
import math
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# -----------------------------
# Utilities: SHA-256 truncation
# -----------------------------

def sha256_trunc_bits(data: bytes, t: int) -> int:
    """
    Return MSB_t(SHA256(data)) as an integer in [0, 2^t).
    We take the most significant t bits of the 256-bit digest.
    """
    digest = hashlib.sha256(data).digest()  # 32 bytes = 256 bits
    x = int.from_bytes(digest, "big")
    return x >> (256 - t)

def encode_seed_counter(seed64: int, counter: int) -> bytes:
    """
    Reproducible input encoding for birthday-table attack:
    m = seed||counter, both fixed-width big-endian.
    """
    return seed64.to_bytes(8, "big") + counter.to_bytes(8, "big")

def encode_state(x: int, t: int) -> bytes:
    """
    Encoding for Pollard rho state to message:
    message = b"rho" || t(2 bytes) || x (ceil(t/8) bytes).
    Including t prevents accidental cross-t collisions in experiments.
    """
    x_bytes_len = (t + 7) // 8
    return b"rho" + t.to_bytes(2, "big") + x.to_bytes(x_bytes_len, "big")

def f_state(x: int, t: int) -> int:
    """
    Iteration function f: {0,1}^t -> {0,1}^t
    f(x) = H_t( encode_state(x) )
    """
    return sha256_trunc_bits(encode_state(x, t), t)

# -----------------------------
# Attack 1: Birthday table
# -----------------------------

@dataclass
class BirthdayResult:
    t: int
    queries: int
    entries: int
    elapsed_sec: float

def birthday_collision(t: int, seed64: int, max_queries: int = 10_000_000) -> BirthdayResult:
    """
    Find first collision in H_t(seed||i) using a hash table.
    Returns number of queries and number of stored entries.
    """
    seen: Dict[int, int] = {}
    start = time.perf_counter()
    for i in range(max_queries):
        m = encode_seed_counter(seed64, i)
        y = sha256_trunc_bits(m, t)
        prev = seen.get(y)
        if prev is not None:
            # Collision found: seed||prev and seed||i map to same y (with overwhelming probability prev != i)
            elapsed = time.perf_counter() - start
            return BirthdayResult(t=t, queries=i + 1, entries=len(seen), elapsed_sec=elapsed)
        seen[y] = i
    raise RuntimeError(f"Birthday attack did not find collision within max_queries={max_queries}")

# -----------------------------
# Attack 2: Pollard Rho (Brent)
# -----------------------------

@dataclass
class RhoResult:
    t: int
    steps_f_calls: int
    mu: int
    lam: int
    elapsed_sec: float

def brent_rho_mu_lam(t: int, x0: int, max_steps: int = 50_000_000) -> Tuple[int, int, int]:
    """
    Brent cycle detection for x_{k+1}=f(x_k).
    Returns (mu, lam, f_calls), where f_calls is count of f evaluations.
    """
    f_calls = 0

    power = 1
    lam = 1
    tortoise = x0
    hare = f_state(x0, t); f_calls += 1

    steps = 0
    while tortoise != hare:
        if steps >= max_steps:
            raise RuntimeError("Brent rho exceeded max_steps while searching for collision.")
        if power == lam:
            tortoise = hare
            power <<= 1
            lam = 0
        hare = f_state(hare, t); f_calls += 1
        lam += 1
        steps += 1

    # Find mu
    mu = 0
    tortoise = x0
    hare = x0
    # Advance hare by lam
    for _ in range(lam):
        hare = f_state(hare, t); f_calls += 1
    while tortoise != hare:
        tortoise = f_state(tortoise, t); f_calls += 1
        hare = f_state(hare, t); f_calls += 1
        mu += 1
        if mu >= max_steps:
            raise RuntimeError("Brent rho exceeded max_steps while finding mu.")

    return mu, lam, f_calls

def rho_collision(t: int, seed64: int, max_steps: int = 50_000_000) -> RhoResult:
    """
    Pollard rho style collision search on the t-bit state space using Brent.
    We count f-calls as the main work metric (each f-call does one SHA-256).
    """
    # Deterministic start state from seed (map 64-bit seed into t-bit state)
    x0 = seed64 & ((1 << t) - 1)

    start = time.perf_counter()
    mu, lam, f_calls = brent_rho_mu_lam(t=t, x0=x0, max_steps=max_steps)
    elapsed = time.perf_counter() - start

    # f_calls is a good proxy for "queries" (SHA-256 calls).
    return RhoResult(t=t, steps_f_calls=f_calls, mu=mu, lam=lam, elapsed_sec=elapsed)

# -----------------------------
# Experiment runner + plots
# -----------------------------

@dataclass
class Aggregated:
    t: int
    method: str
    trials: int
    mean_queries: float
    std_queries: float
    mean_time_sec: float
    std_time_sec: float
    mean_memory_entries: Optional[float]  # Only for birthday
    std_memory_entries: Optional[float]

def mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)

def run_experiments(
    t_values: List[int],
    trials_per_t: int = 200,
    out_csv: str = "results.csv",
    seed_base: int = 123456789,
) -> List[Aggregated]:
    """
    Runs both methods for each t and aggregates results.
    Writes per-trial results to CSV for reproducibility.
    """
    rows = []
    for t in t_values:
        for trial in range(trials_per_t):
            # Make deterministic, distinct seeds per (t, trial)
            seed64 = (seed_base + (t * 1_000_000) + trial) & ((1 << 64) - 1)

            # Birthday
            b = birthday_collision(t=t, seed64=seed64)
            rows.append({
                "t": t,
                "trial": trial,
                "method": "birthday",
                "queries": b.queries,
                "time_sec": b.elapsed_sec,
                "memory_entries": b.entries,
            })

            # Rho
            r = rho_collision(t=t, seed64=seed64)
            rows.append({
                "t": t,
                "trial": trial,
                "method": "rho_brent",
                "queries": r.steps_f_calls,
                "time_sec": r.elapsed_sec,
                "memory_entries": "",  # N/A
                "mu": r.mu,
                "lam": r.lam,
            })

        print(f"Done t={t} ({trials_per_t} trials)")

    # Write CSV
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Aggregate
    aggs: List[Aggregated] = []
    for t in t_values:
        for method in ["birthday", "rho_brent"]:
            qs = [float(r["queries"]) for r in rows if r["t"] == t and r["method"] == method]
            ts = [float(r["time_sec"]) for r in rows if r["t"] == t and r["method"] == method]
            mq, sq = mean_std(qs)
            mt, st = mean_std(ts)
            if method == "birthday":
                mem = [float(r["memory_entries"]) for r in rows if r["t"] == t and r["method"] == method]
                mm, sm = mean_std(mem)
            else:
                mm = sm = None
            aggs.append(Aggregated(
                t=t, method=method, trials=trials_per_t,
                mean_queries=mq, std_queries=sq,
                mean_time_sec=mt, std_time_sec=st,
                mean_memory_entries=mm, std_memory_entries=sm
            ))
    return aggs

def plot_results(aggs: List[Aggregated], out1: str, out2: str) -> None:
    """
    Create:
      1) log2(mean queries) vs t
      2) mean memory entries vs t (birthday only)
    """
    import matplotlib.pyplot as plt

    t_values = sorted(set(a.t for a in aggs))
    methods = sorted(set(a.method for a in aggs))

    # Plot 1: log2(mean queries) vs t
    plt.figure()
    for method in methods:
        xs = []
        ys = []
        for t in t_values:
            a = next(x for x in aggs if x.t == t and x.method == method)
            xs.append(t)
            ys.append(math.log2(a.mean_queries))
        plt.plot(xs, ys, marker="o", label=method)
    # Reference slope 0.5 line: y = 0.5 t + c, choose c to pass through first birthday point
    a0 = next(x for x in aggs if x.t == t_values[0] and x.method == "birthday")
    c = math.log2(a0.mean_queries) - 0.5 * t_values[0]
    ref = [0.5 * t + c for t in t_values]
    plt.plot(t_values, ref, linestyle="--", label="reference: 0.5*t + c")

    plt.xlabel("t (truncated output bits)")
    plt.ylabel("log2(mean #queries)")
    plt.title("Collision search cost vs truncation bits (SHA-256 truncated)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out1, dpi=200)

    # Plot 2: memory entries vs t (birthday only)
    plt.figure()
    xs = []
    ys = []
    for t in t_values:
        a = next(x for x in aggs if x.t == t and x.method == "birthday")
        xs.append(t)
        ys.append(a.mean_memory_entries)
    plt.plot(xs, ys, marker="o", label="birthday memory (entries)")
    plt.xlabel("t (truncated output bits)")
    plt.ylabel("mean stored entries until first collision")
    plt.title("Birthday-table memory growth vs truncation bits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out2, dpi=200)

def main():
    # Adjustable experiment plan
    t_values = list(range(20, 32, 2))  # 20,22,...,30
    trials_per_t = 200                 # reduce to 100 if too slow

    aggs = run_experiments(
        t_values=t_values,
        trials_per_t=trials_per_t,
        out_csv="results.csv",
        seed_base=0xC0FFEE,
    )

    # Print a compact summary table
    print("\nSummary (mean queries ± std):")
    for t in t_values:
        b = next(x for x in aggs if x.t == t and x.method == "birthday")
        r = next(x for x in aggs if x.t == t and x.method == "rho_brent")
        print(
            f"t={t:2d} | birthday: {b.mean_queries:10.1f} ± {b.std_queries:8.1f} "
            f"| rho: {r.mean_queries:10.1f} ± {r.std_queries:8.1f}"
        )

    plot_results(
        aggs,
        out1="plot_logQ_vs_t.png",
        out2="plot_memory_vs_t.png",
    )
    print("\nWrote: results.csv, plot_logQ_vs_t.png, plot_memory_vs_t.png")

if __name__ == "__main__":
    main()
