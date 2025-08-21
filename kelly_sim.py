#!/usr/bin/env python3
import argparse
import math
import numpy as np
import csv

def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds (including stake)."""
    if american_odds == 0:
        raise ValueError("American odds cannot be 0.")
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    else:
        return 1.0 + 100.0 / abs(american_odds)

def kelly_fraction(p: float, b: float) -> float:
    """Full Kelly fraction for a binary bet with probability p and net odds b (decimal-1)."""
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)  # no negative betting

def expected_log_growth(f: float, p: float, b: float) -> float:
    """Expected log growth per bet for fraction f."""
    if f <= -1.0 or f >= 1.0:
        return float('-inf')
    q = 1.0 - p
    if f == 0.0:
        return 0.0
    if 1.0 - f <= 0 or 1.0 + f * b <= 0:
        return float('-inf')
    return p * math.log(1.0 + f * b) + q * math.log(1.0 - f)

def simulate_paths(n_sims: int, n_bets: int, p: float, b: float, f: float,
                   bankroll0: float = 1.0, ruin_threshold: float = 0.0,
                   seed: int = 42, sample_paths: int = 0):
    """Simulate bankroll paths; return aggregate stats and optional sample trajectories."""
    rng = np.random.default_rng(seed)
    outcomes = rng.random((n_sims, n_bets)) < p  # True = win

    win_mult = 1.0 + f * b
    lose_mult = 1.0 - f
    if win_mult <= 0 or lose_mult <= 0:
        raise ValueError("Invalid f leading to non-positive bankroll multiplier. Reduce f.")

    log_win = math.log(win_mult)
    log_lose = math.log(lose_mult)

    log_bank = np.full((n_sims,), math.log(bankroll0), dtype=float)
    max_bank = np.full((n_sims,), bankroll0, dtype=float)
    max_dd = np.zeros((n_sims,), dtype=float)
    ruin = np.zeros((n_sims,), dtype=bool)

    samples = []
    if sample_paths > 0:
        sample_indices = np.arange(min(sample_paths, n_sims))
        sample_trajs = np.full((len(sample_indices), n_bets + 1), bankroll0, dtype=float)

    for t in range(n_bets):
        wins = outcomes[:, t]
        log_bank += np.where(wins, log_win, log_lose)
        bank = np.exp(log_bank)

        max_bank = np.maximum(max_bank, bank)
        dd = 1.0 - bank / max_bank
        max_dd = np.maximum(max_dd, dd)
        ruin = np.logical_or(ruin, bank <= ruin_threshold * bankroll0)

        if sample_paths > 0:
            sample_trajs[:, t+1] = bank[:len(sample_indices)]

    terminal = np.exp(log_bank)
    stats = {
        "terminal_mean": float(np.mean(terminal)),
        "terminal_median": float(np.median(terminal)),
        "terminal_p10": float(np.percentile(terminal, 10)),
        "terminal_p1": float(np.percentile(terminal, 1)),
        "log_growth_per_bet": expected_log_growth(f, p, b),
        "ruin_prob": float(np.mean(ruin)),
        "max_dd_median": float(np.median(max_dd)),
        "max_dd_p90": float(np.percentile(max_dd, 90)),
    }
    if sample_paths > 0:
        for i in range(sample_trajs.shape[0]):
            samples.append(sample_trajs[i].tolist())
        stats["sample_paths"] = samples
    return stats

def parse_args():
    ap = argparse.ArgumentParser(description="Kelly Criterion Betting Simulator")
    sub = ap.add_subparsers(dest="mode", required=True)

    def add_common(a):
        a.add_argument("--bankroll0", type=float, default=1000.0, help="Starting bankroll")
        a.add_argument("--bets", type=int, default=200, help="Number of bets to simulate")
        a.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo paths")
        a.add_argument("--seed", type=int, default=42, help="Random seed")
        a.add_argument("--grid", type=str, default="0.25,0.5,0.75,1.0",
                       help="Comma-separated Kelly multipliers to evaluate")
        a.add_argument("--ruin-threshold", type=float, default=0.0,
                       help="Ruin if bankroll ≤ threshold × initial")
        a.add_argument("--samples", type=int, default=0,
                       help="Number of sample paths to record")
        a.add_argument("--export", type=str, default=None,
                       help="CSV filepath to export summary table")
        a.add_argument("--plot", action="store_true",
                       help="Plot bankroll sample paths and growth curve")

    a1 = sub.add_parser("binary", help="Binary bet with probability p and net odds b = decimal-1")
    add_common(a1)
    a1.add_argument("--p", type=float, help="Win probability (0<p<1)")
    a1.add_argument("--decimal-odds", type=float, help="Decimal odds including stake")
    a1.add_argument("--american-odds", type=float, help="American odds")
    a1.add_argument("--wins", type=int, help="Use empirical wins to estimate p")
    a1.add_argument("--losses", type=int, help="Use empirical losses to estimate p")
    a1.add_argument("--prior", type=str, default="jeffreys",
                    help="Prior: 'jeffreys' (0.5,0.5) or 'uniform' (1,1)")

    a2 = sub.add_parser("sports", help="Sports moneyline odds with subjective probability p")
    add_common(a2)
    a2.add_argument("--p", type=float, required=True, help="Your subjective win probability")
    a2.add_argument("--american-odds", type=float, required=True, help="Moneyline odds")
    return ap.parse_args()

def estimate_p_from_record(wins: int, losses: int, prior: str = "jeffreys") -> float:
    a0, b0 = (0.5, 0.5) if prior == "jeffreys" else (1.0, 1.0)
    a_post = a0 + wins
    b_post = b0 + losses
    return a_post / (a_post + b_post)

def run_mode(p: float, b: float, args):
    f_full = kelly_fraction(p, b)
    grid = [float(x.strip()) for x in args.grid.split(",") if x.strip()]
    results = []
    rows_for_csv = []
    growth_points = []

    for mult in grid:
        f = mult * f_full
        if f < 0:
            f = 0.0
        if f >= 1.0:
            f = 0.999999
        stats = simulate_paths(
            n_sims=args.sims, n_bets=args.bets, p=p, b=b, f=f,
            bankroll0=args.bankroll0, ruin_threshold=args.ruin_threshold,
            seed=args.seed, sample_paths=args.samples
        )
        results.append((mult, f, stats))
        rows_for_csv.append({
            "mult": mult,
            "f": f,
            "E_log_growth": stats["log_growth_per_bet"],
            "ruin_prob": stats["ruin_prob"],
            "median_TW": stats["terminal_median"],
            "p10_TW": stats["terminal_p10"],
            "p1_TW": stats["terminal_p1"],
            "med_maxDD": stats["max_dd_median"],
            "p90_maxDD": stats["max_dd_p90"],
        })
        growth_points.append((f, stats["log_growth_per_bet"]))

    print("\nKelly Criterion Betting Simulator")
    print(f"Inputs: p={p:.4f}, decimal_odds={1+b:.4f} (b={b:.4f}), full_kelly={f_full:.4f}")
    print(f"Simulating {args.sims} paths x {args.bets} bets, bankroll0={args.bankroll0}, ruin_threshold={args.ruin_threshold}")
    headers = ["mult", "f", "E[ln(1+R)]", "ruin_prob", "median_TW", "p10_TW", "p1_TW", "med_maxDD", "p90_maxDD"]
    print("\t".join(headers))
    for mult, f, s in results:
        print(f"{mult:.2f}\t{f:.4f}\t{s['log_growth_per_bet']:.6f}\t{s['ruin_prob']:.4f}\t"
              f"{s['terminal_median']:.2f}\t{s['terminal_p10']:.2f}\t{s['terminal_p1']:.2f}\t"
              f"{s['max_dd_median']:.2%}\t{s['max_dd_p90']:.2%}")

    if args.samples > 0 and results:
        sp = results[0][2]["sample_paths"][0]
        print("\nFirst sample path (first grid mult) bankroll trajectory (first 10 pts):",
              [round(x,2) for x in sp[:10]])

    # Export to CSV
    if args.export:
        fieldnames = ["mult", "f", "E_log_growth", "ruin_prob",
                      "median_TW", "p10_TW", "p1_TW", "med_maxDD", "p90_maxDD"]
        with open(args.export, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_for_csv:
                w.writerow(r)
        print(f"\n[Saved] Summary table → {args.export}")

    # Plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[Plot skipped] matplotlib not installed. Install with: pip install matplotlib")
        else:
            if args.samples > 0 and results and "sample_paths" in results[0][2]:
                samples = results[0][2]["sample_paths"]
                x = list(range(len(samples[0])))
                plt.figure()
                for s in samples:
                    plt.plot(x, s, linewidth=1)
                plt.title("Sample Bankroll Paths (first grid multiplier)")
                plt.xlabel("Bet #")
                plt.ylabel("Bankroll")
                plt.tight_layout()

            if growth_points:
                f_list = [p[0] for p in growth_points]
                g_list = [p[1] for p in growth_points]
                plt.figure()
                plt.plot(f_list, g_list, marker="o")
                plt.title("Expected Log-Growth vs Bet Fraction")
                plt.xlabel("Bet fraction f")
                plt.ylabel("E[ln(1+R)] per bet")
                plt.tight_layout()

            plt.show()

def main():
    args = parse_args()
    if args.mode == "binary":
        if args.p is not None:
            p = args.p
        elif args.wins is not None and args.losses is not None:
            p = estimate_p_from_record(args.wins, args.losses, args.prior)
        else:
            raise SystemExit("Provide --p or both --wins and --losses.")
        if args.decimal_odds is not None:
            decimal = args.decimal_odds
        elif args.american_odds is not None:
            decimal = american_to_decimal(args.american_odds)
        else:
            raise SystemExit("Provide --decimal-odds or --american-odds.")
        b = decimal - 1.0
        run_mode(p, b, args)
    elif args.mode == "sports":
        p = args.p
        decimal = american_to_decimal(args.american_odds)
        b = decimal - 1.0
        run_mode(p, b, args)

if __name__ == "__main__":
    main()
