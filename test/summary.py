#!/usr/bin/env python3
"""
summary.py - Evaluate result summary

Reads JSON result files from test/result/ and computes per-metric averages
and >=1 counts.

Usage:
    python test/summary.py                          # Process all JSON in test/result/
    python test/summary.py test/result/xxx.json     # Process specific file
"""

import json
import os
import sys
import glob
from collections import defaultdict

METRICS = ["correctness", "faithfulness", "answer_relevance", "context_relevance"]
METADATA_KEYS = {"id", "query", "answer"}
FT_NA = {"faithfulness", "context_relevance"}  # N/A metrics for Fast-Track


def resolve_path(p: str) -> str:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(p):
        p = os.path.join(_root, p)
    return os.path.normpath(p)


def get_field_names(entry: dict) -> list:
    """Identify evaluation fields (exclude id/query/answer metadata)"""
    return [k for k in entry if k not in METADATA_KEYS]


def field_is_valid(entry: dict, field: str) -> tuple:
    """Check if a field is valid, return (valid, reason)"""
    f = entry.get(field)
    if not f:
        return False, f"field [{field}] missing"

    if not f.get("response"):
        return False, f"{field}.response is empty"

    # loop=-2 Agent fallback: reasoning intentionally empty (forced scores) -> valid
    is_fallback = (field == "Agent" and f.get("loop_counter") == -2)
    if not is_fallback:
        ds = f.get("deepseek", {})
        glm = f.get("GLM", {})
        if not ds.get("reasoning"):
            return False, f"{field}.deepseek.reasoning is empty"
        if not glm.get("reasoning"):
            return False, f"{field}.GLM.reasoning is empty"

    return True, ""


def collect_scores(field_data: dict, metrics: list) -> dict:
    """Collect DeepSeek + GLM scores for given metrics from a field"""
    scores = {m: [] for m in metrics}
    for judge_key in ("deepseek", "GLM"):
        judge = field_data.get(judge_key, {})
        s = judge.get("scores", {})
        for m in metrics:
            val = s.get(m)
            if val is not None:
                scores[m].append(val)
    return scores


def calc_summary(scores: dict) -> dict:
    """Compute average and >=1 stats for each metric"""
    result = {}
    for metric, vals in scores.items():
        if not vals:
            result[metric] = {"avg": "N/A", "ge1_num": "N/A", "ge1_pct": "N/A", "n": 0}
        else:
            n = len(vals)
            avg = sum(vals) / n
            ge1 = sum(1 for v in vals if v >= 1)
            result[metric] = {
                "avg": round(avg, 4),
                "ge1_num": ge1,
                "ge1_pct": round(ge1 / n * 100, 1),
                "n": n,
            }
    return result


def print_separator(char="=", width=72):
    print(char * width)


def print_table(summary: dict, show_na: set = None):
    """Print a formatted metrics table"""
    if show_na is None:
        show_na = set()
    header = f"{'Metric':<22} {'Avg':>8} {'>=1 Count':>10} {'>=1 %':>8}"
    print(header)
    print("-" * len(header))
    for m in METRICS:
        if m in show_na:
            print(f"  {m:<20} {'N/A':>8} {'N/A':>10} {'N/A':>8}")
        else:
            s = summary.get(m, {})
            if s.get("n", 0) == 0:
                print(f"  {m:<20} {'N/A':>8} {'N/A':>10} {'N/A':>8}")
            else:
                print(f"  {m:<20} {s['avg']:>8.4f} {s['ge1_num']:>10} {s['ge1_pct']:>7.1f}%")


# ==================== Processors ====================


def process_generic_field(entries: list, field: str):
    """Process a generic field (RAG / no_fix / baseline, etc.)"""
    all_scores = {m: [] for m in METRICS}
    for entry in entries:
        fd = entry.get(field)
        if not fd:
            continue
        scores = collect_scores(fd, METRICS)
        for m in METRICS:
            all_scores[m].extend(scores[m])

    summary = calc_summary(all_scores)
    print(f"  ({len(entries)} valid entries)")
    print_table(summary)


def process_agent(entries: list):
    """Process Agent field: split by loop_counter"""
    groups = {"fast_track": [], "rag_0": [], "rag_1": [], "fallback": []}

    for entry in entries:
        fd = entry.get("Agent")
        if not fd:
            continue
        lc = fd.get("loop_counter")
        if lc == -1:
            groups["fast_track"].append(entry)
        elif lc == -2:
            groups["fallback"].append(entry)
        elif lc == 0:
            groups["rag_0"].append(entry)
        else:
            groups["rag_1"].append(entry)

    n_fast = len(groups["fast_track"])
    n_rag0 = len(groups["rag_0"])
    n_rag1 = len(groups["rag_1"])
    n_rag = n_rag0 + n_rag1
    n_fb = len(groups["fallback"])
    total_agent = n_fast + n_rag + n_fb
    fb_pct = round(n_fb / total_agent * 100, 1) if total_agent else 0

    # Fast-Track (correctness + answer_relevance only)
    if n_fast > 0:
        print(f"\n  [Fast-Track] (loop=-1, no RAG): {n_fast} entries")
        ft_scores = {m: [] for m in METRICS}
        for entry in groups["fast_track"]:
            fd = entry.get("Agent", {})
            scores = collect_scores(fd, METRICS)
            for m in ("correctness", "answer_relevance"):
                ft_scores[m].extend(scores[m])
        ft_summary = calc_summary(ft_scores)
        print_table(ft_summary, show_na=FT_NA)

    # RAG (all 4 metrics), with loop=0 vs loop=1 breakdown
    if n_rag > 0:
        rag0_pct = round(n_rag0 / n_rag * 100, 1) if n_rag else 0
        rag1_pct = round(n_rag1 / n_rag * 100, 1) if n_rag else 0
        print(f"\n  [RAG] (loop=0/1): {n_rag} entries "
              f"(loop=0: {n_rag0} / {rag0_pct}%, loop=1: {n_rag1} / {rag1_pct}%)")
        rag_scores = {m: [] for m in METRICS}
        for entry in groups["rag_0"] + groups["rag_1"]:
            fd = entry.get("Agent", {})
            scores = collect_scores(fd, METRICS)
            for m in METRICS:
                rag_scores[m].extend(scores[m])
        rag_summary = calc_summary(rag_scores)
        print_table(rag_summary)

    # Fallback (proportion only)
    if n_fb > 0:
        print(f"\n  [Fallback] (loop=-2): {n_fb} entries ({fb_pct}% of Agent)")

    # Summary line
    print(f"  Total: {total_agent} (Fast-Track {n_fast} | RAG {n_rag} | Fallback {n_fb})")


# ==================== Main ====================


def process_file(filepath: str):
    print_separator()
    print(f"  File: {os.path.basename(filepath)}")
    print_separator()

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("  (not a list format, skipped)")
        return

    # ---- validity check ----
    skipped_ids = []
    valid_entries = []

    for entry in data:
        eid = entry.get("id", "?")
        fields = get_field_names(entry)
        skip_this = False

        for field in fields:
            valid, _ = field_is_valid(entry, field)
            if not valid:
                skip_this = True
                break

        if skip_this:
            skipped_ids.append(eid)
        else:
            valid_entries.append(entry)

    total = len(data)
    valid = len(valid_entries)
    print(f"\n  Total entries: {total} | Valid: {valid} | Skipped: {len(skipped_ids)}")
    if skipped_ids:
        print(f"  Skipped IDs: {skipped_ids}")

    if not valid_entries:
        print("  No valid entries.")
        return

    field_names = get_field_names(valid_entries[0])

    for field in field_names:
        print(f"\n  == {field} ==")
        if field == "Agent":
            process_agent(valid_entries)
        else:
            process_generic_field(valid_entries, field)


def main():
    result_dir = resolve_path("./test/result/")

    if len(sys.argv) > 1:
        paths = [resolve_path(sys.argv[1])]
    else:
        paths = sorted(glob.glob(os.path.join(result_dir, "*.json")))

    if not paths:
        print(f"No JSON files found in: {result_dir}")
        sys.exit(1)

    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue
        process_file(p)


if __name__ == "__main__":
    main()
