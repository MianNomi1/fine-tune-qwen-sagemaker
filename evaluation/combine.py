#!/usr/bin/env python3
"""Print the before/after comparison from reports/base.json and reports/tuned.json.

Run after evaluating both models:
    python combine.py
Writes reports/before_after.md and prints the headline tables.
"""
import json
from pathlib import Path

b = json.load(open("reports/base.json"))["report"]
t = json.load(open("reports/tuned.json"))["report"]

lines = []
lines.append("# Horizon Bank QLoRA: before vs after\n")

lines.append("## Trained behaviour & facts (the headline)\n")
lines.append("| Metric | Base | Tuned | Delta |")
lines.append("|---|---|---|---|")
for k, label in [("pass_pct", "Overall pass"), ("facts_pct", "Fact/behaviour coverage"),
                 ("numbers_pct", "Exact-number accuracy"), ("currency_ok_pct", "Correct currency (USD)")]:
    bo, to = b["overall"][k], t["overall"][k]
    sign = "+" if to - bo >= 0 else ""
    lines.append(f"| {label} | {bo}% | {to}% | {sign}{round(to - bo, 1)} |")

lines.append("\n## Per category — trained items (overall pass %)\n")
lines.append("| Category | Base | Tuned |")
lines.append("|---|---|---|")
for cat in t["by_category"]:
    bo = b["by_category"].get(cat, {}).get("pass_pct", 0.0)
    to = t["by_category"][cat]["pass_pct"]
    lines.append(f"| {cat} | {bo}% | {to}% |")

lines.append("\n## Held-out facts — the RAG lesson\n")
lines.append("These facts were NEVER in training. Both models should score LOW: this is the")
lines.append("evidence that facts the model hasn't memorised must come from retrieval (RAG),")
lines.append("not from the weights.\n")
lines.append("| Metric | Base | Tuned |")
lines.append("|---|---|---|")
lines.append(f"| Held-out fact pass | {b['held_out']['pass_pct']}% | {t['held_out']['pass_pct']}% |")

out = "\n".join(lines) + "\n"
Path("reports/before_after.md").write_text(out)
print(out)
print("Wrote reports/before_after.md")
