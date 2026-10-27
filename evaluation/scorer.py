#!/usr/bin/env python3
"""Pure scoring functions for the before/after evaluation. No ML deps.

Each test row (from data/test.jsonl) carries scoring metadata:
  required_facts   - substrings a correct answer must contain (word-ish match)
  required_numbers - exact figures a correct answer must contain (whole-number match)
  expect_refusal   - True if the right behaviour is to politely decline
  held_out         - True if this fact was NEVER in training (the RAG-lesson probe)
  category         - topic label, for per-category breakdown

This scorer is intentionally strict so the before/after gap is credible:
  * numbers must match as WHOLE tokens ("8" does not match "$80" or "2018");
  * a USD bank answering with a foreign currency symbol (£, €) FAILS;
  * a refusal must actually decline, not merely mention the app inside a made-up answer.

score_one() grades one answer against one row. aggregate() rolls results into overall,
per-category, and a separate "held-out facts" view.
"""
from __future__ import annotations

import re

FOREIGN_CURRENCY = ["£", "€", "gbp", "eur", "pounds", "euros", "rupees", "₹", "¥"]
REFUSAL_MARKERS = ["can't", "cannot", "can not", "only help", "unable to",
                   "not able to", "outside what i can", "i'm the horizon bank assistant",
                   "check your balance in", "in the horizon mobile app", "online banking"]
# Phrases that mean the model wrongly refused a legitimate question.
WRONG_REFUSAL = ["i can't help", "cannot help", "i'm unable", "unable to assist",
                 "can't assist with that", "i do not have access"]


def _has_number(answer_low: str, number: str) -> bool:
    """True only if `number` appears as a standalone figure, not embedded in a bigger one.

    "8" matches "8 USD" and "is 8." but NOT "$80", "2018", or "0.85".
    Commas in the expected value are kept (e.g. "1,500").
    """
    esc = re.escape(number.lower())
    # Reject when the expected number is part of a BIGGER number:
    #  - preceded by a digit/comma/decimal digit  (e.g. "$80" for "8", "12,000" for "2,000")
    #  - followed by a digit, a comma+digit, or a decimal (".5"); a bare sentence period is fine.
    pattern = rf"(?<![\d.,]){esc}(?!\d)(?!,\d)(?!\.\d)"
    return re.search(pattern, answer_low) is not None


def _uses_foreign_currency(answer_low: str) -> bool:
    return any(sym in answer_low for sym in FOREIGN_CURRENCY)


def score_one(answer: str, row: dict) -> dict:
    low = answer.lower()
    held_out = bool(row.get("held_out"))
    category = row["category"]

    # --- refusal rows: correct behaviour is to decline + redirect ---
    if row.get("expect_refusal"):
        refused = any(m in low for m in REFUSAL_MARKERS)
        return {"category": category, "held_out": held_out, "passed": refused,
                "facts_ok": refused, "numbers_ok": True, "currency_ok": True,
                "refusal_ok": refused}

    facts_ok = all(f.lower() in low for f in row.get("required_facts", []))
    numbers_ok = all(_has_number(low, n) for n in row.get("required_numbers", []))
    currency_ok = not _uses_foreign_currency(low)          # USD bank must not use £/€
    wrongly_refused = any(m in low for m in WRONG_REFUSAL)

    passed = facts_ok and numbers_ok and currency_ok and not wrongly_refused
    return {"category": category, "held_out": held_out, "passed": passed,
            "facts_ok": facts_ok, "numbers_ok": numbers_ok, "currency_ok": currency_ok,
            "refusal_ok": not wrongly_refused}


def _pct(items, key):
    return round(100 * sum(r[key] for r in items) / len(items), 1) if items else 0.0


def aggregate(results: list[dict]) -> dict:
    # Separate the held-out probes from the trained items: they answer two different
    # questions ("did training work?" vs "does it hallucinate on unseen facts?").
    trained = [r for r in results if not r.get("held_out")]
    held = [r for r in results if r.get("held_out")]

    def block(items):
        return {"n": len(items),
                "pass_pct": _pct(items, "passed"),
                "facts_pct": _pct(items, "facts_ok"),
                "numbers_pct": _pct(items, "numbers_ok"),
                "currency_ok_pct": _pct(items, "currency_ok")}

    by_cat = {}
    for cat in sorted({r["category"] for r in trained}):
        by_cat[cat] = block([r for r in trained if r["category"] == cat])

    return {
        "overall": block(trained),            # trained facts only -> the headline before/after
        "by_category": by_cat,
        "held_out": block(held),              # the RAG-lesson probe (expect LOW pass for both models)
    }
