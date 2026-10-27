#!/usr/bin/env python3
"""Run a model over data/test.jsonl, score it, and write JSON + CSV reports.

Run once per model (base and tuned), then compare with combine.py:
    python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --label base
    python evaluate.py --model ../merged                   --label tuned

Works on CPU (slow but $0) or GPU. The base model id downloads from Hugging Face;
the tuned model is a local folder - from_pretrained accepts either.
"""
import argparse, csv, json
from pathlib import Path
from scorer import score_one, aggregate

SYS = ("You are the Horizon Bank virtual assistant. Answer using only Horizon Bank's published "
       "policy. Be concise and accurate. For loan eligibility say the result is indicative and "
       "not a final offer. If a question is outside Horizon Bank banking, politely decline.")


def generate_all(model_path, rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    # Load plainly on CPU in fp32. Do NOT use device_map="auto" on a laptop - it can
    # offload layers to disk ("meta device") when RAM is tight, which is slow and buggy.
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    answers = []
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        # tokenize=True + return_dict=True gives a dict with input_ids we can pass explicitly.
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_new_tokens=220, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        answers.append(tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True))
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--test", default="../data/test.jsonl")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.test) if l.strip()]
    answers = generate_all(a.model, rows)
    results = [score_one(ans, r) for ans, r in zip(answers, rows)]
    report = aggregate(results)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    detail = [{"q": next(m["content"] for m in r["messages"] if m["role"] == "user"),
               "answer": ans, **res} for r, ans, res in zip(rows, answers, results)]
    json.dump({"label": a.label, "report": report, "detail": detail},
              open(f"{a.out}/{a.label}.json", "w"), indent=2)
    with open(f"{a.out}/{a.label}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "held_out", "passed", "facts_ok", "numbers_ok", "currency_ok", "question"])
        for r, res in zip(rows, results):
            q = next(m["content"] for m in r["messages"] if m["role"] == "user")
            w.writerow([res["category"], res.get("held_out", False), res["passed"],
                        res["facts_ok"], res["numbers_ok"], res.get("currency_ok", True), q])

    o = report["overall"]
    print(f"\n{a.label}: TRAINED pass {o['pass_pct']}%  facts {o['facts_pct']}%  "
          f"numbers {o['numbers_pct']}%  currency_ok {o['currency_ok_pct']}%")
    for cat, v in report["by_category"].items():
        print(f"   {cat:16} pass {v['pass_pct']:5}%  numbers {v['numbers_pct']:5}%  (n={v['n']})")
    h = report["held_out"]
    print(f"   HELD-OUT facts (never trained): pass {h['pass_pct']}%  (n={h['n']})  "
          f"<- expect LOW; this is the RAG lesson")


if __name__ == "__main__":
    main()
