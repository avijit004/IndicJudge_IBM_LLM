"""
Compare the judge's actual scores with the PT1-PT4 targets.

    python perturbation_experiment/summarize_results.py                 # all results files
    python perturbation_experiment/summarize_results.py --judge-tag ministral8b

Writes perturbation_experiment/results/summary_by_prompt.csv and prints a table.
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturbation_prompts import PERTURBATION_TARGETS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-tag", default="*", help="Prefix of results files (default: all judges).")
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{args.judge_tag}_*_perturbation_results.csv")))
    if not files:
        print("No results files found. Run judge_perturbations.py first.")
        return 1

    groups = defaultdict(list)
    for path in files:
        tag = os.path.basename(path).split("_")[0]
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                groups[(tag, r["prompt_id"])].append(r)

    out_rows = []
    header = f"{'judge':<14}{'prompt':<7}{'n':>5}{'mean RQ':>9}{'target':>8}{'in-target':>11}{'ans correct':>13}{'expected':>10}{'intent':>8}"
    print(header)
    print("-" * len(header))
    for (tag, pid), rows in sorted(groups.items()):
        n = len(rows)
        rq = [int(r["reasoning_quality"]) for r in rows]
        correct = sum(r["answer_correct"] == "True" for r in rows) / n
        intent = sum(r["follows_prompt_intent"] == "True" for r in rows) / n
        t = PERTURBATION_TARGETS[pid]
        hit = sum(t["rq_min"] <= x <= t["rq_max"] for x in rq) / n
        target = f"{t['rq_min']}" if t["rq_min"] == t["rq_max"] else f"{t['rq_min']}-{t['rq_max']}"
        expected = {True: "true", False: "false", None: "either"}[t["expected_answer_correct"]]
        dist = {k: rq.count(k) for k in range(6)}
        out_rows.append({"judge": tag, "prompt_id": pid, "n": n, "mean_reasoning_quality": round(sum(rq) / n, 2),
                         "target": target, "pct_rq_in_target": round(100 * hit, 1),
                         "pct_answer_correct": round(100 * correct, 1), "expected_answer_correct": expected,
                         "pct_follows_intent": round(100 * intent, 1),
                         "rq_distribution_0_to_5": "/".join(str(dist[k]) for k in range(6))})
        print(f"{tag:<14}{pid:<7}{n:>5}{sum(rq)/n:>9.2f}{target:>8}{100*hit:>10.1f}%{100*correct:>12.1f}%{expected:>10}{100*intent:>7.1f}%")

    out = os.path.join(RESULTS_DIR, "summary_by_prompt.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
