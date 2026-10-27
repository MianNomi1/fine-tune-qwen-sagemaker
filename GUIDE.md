# Hands-on: QLoRA fine-tuning of Qwen2.5 on Amazon SageMaker (end to end, under $5)

This guide teaches fine-tune an open-source **Qwen2.5-Instruct** model with **QLoRA**
on **Amazon SageMaker**, then prove the result with a **rigorous, numeric before/after
evaluation**. You write the code; this document explains every piece, why it exists, and how
to verify it. The whole exercise is designed to cost **under $5** of real AWS spend.

You are building a grounded **Horizon Bank** support assistant. A stock Qwen2.5 has never seen
Horizon Bank's fees, loan policy or product rules, so it guesses. After fine-tuning it answers
with the bank's exact figures and the right "indicative, not a final offer" discipline — and
you will measure that difference in numbers, not impressions.

> Reference data already in this folder: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.
> You can regenerate them with the script in Section 3. Everything else you write by following
> this guide.

---

## Table of contents

1. Concepts: QLoRA, and why SageMaker (not Bedrock)
2. Prerequisites and project layout
3. Stage 1 — the dataset (what, why, and the generator code)
4. Stage 2 — the training entrypoint (`train.py`), explained block by block
5. Stage 3 — launching the SageMaker job (`launcher.py`) with hard cost caps
6. Stage 4 — the rigorous before/after evaluation (the core of this lab)
7. The cost table: proving ≤ $5
8. Mandatory cleanup
9. Sources

---

## 1. Concepts

### 1.1 What QLoRA actually is

Full fine-tuning updates every weight in the model — billions of numbers — which needs a lot
of GPU memory. **LoRA** (Low-Rank Adaptation) freezes the original weights and trains small
"adapter" matrices alongside them, so you update a tiny fraction of the parameters. **QLoRA**
goes one step further: it loads the frozen base model in **4-bit** precision (a compression
called NF4), which shrinks its memory footprint dramatically, then trains the LoRA adapters on
top. So:

> QLoRA = (base model quantized to 4-bit) + (train LoRA adapters). That is exactly what "LoRA
> PEFT on a quantized model" means. PEFT = Parameter-Efficient Fine-Tuning, the family LoRA
> belongs to.

Why it matters for you: a 7B model is ~15 GB in 16-bit but only ~5 GB in 4-bit, so QLoRA lets a
single modest GPU fine-tune models that full fine-tuning couldn't touch. The QLoRA paper showed
this reaches full-precision quality while fitting a 65B model on one 48 GB GPU.
(Source: Dettmers et al., arXiv:2305.14314; Hugging Face PEFT quantization guide.)

At the end of training you **merge** the adapters back into the base weights to produce one
normal model you can serve anywhere.

### 1.2 Why SageMaker, and why not Bedrock

This matters because it's the reason the lab changed shape. Two separate things people conflate:

- **Training** (the GPU work where the model learns) — needs a CUDA GPU.
- **Hosting** (serving the finished model) — a separate concern.

**Amazon Bedrock does not let you fine-tune an arbitrary Hugging Face model.** Bedrock's native
fine-tuning only works on a fixed provider list (Amazon Nova/Titan, Anthropic Claude Haiku,
Meta Llama). You cannot point it at Qwen2.5. (Source: AWS "Customize a model with fine-tuning in
Amazon Bedrock".) Bedrock *can host* a custom model you trained elsewhere via Custom Model
Import, but hosting a custom Llama on Bedrock typically needs **Provisioned Throughput**, billed
hourly whether idle or not — which makes a ≤ $5 budget hard to guarantee.

**Amazon SageMaker training is the right tool** because:

- It runs *your own* training code in a managed GPU container, so any Hugging Face model works,
  including Qwen2.5.
- It is **billed per second**, with no minimum and no commitment. The GPU instance exists only
  while your job runs, then SageMaker tears it down. (Source: AWS SageMaker pricing.)
- You never stand up an always-on endpoint, so nothing bills while idle.

That per-second billing is what lets us *promise* the budget. The cost is essentially
(GPU $/hour) × (minutes the job runs), and we cap the minutes.

```
Your laptop (CPU)            SageMaker managed GPU job            Your laptop (CPU)
----------------             -------------------------            ----------------
build dataset           -->  QLoRA training (per-second)     -->  download model
upload data to S3             merge adapters, save model           run before/after eval
launch job                    instance auto-terminates             write numeric reports
```

---

## 2. Prerequisites and project layout

**You need:**

- An AWS account with SageMaker access and a **SageMaker execution role** (in SageMaker Studio
  this is automatic via `sagemaker.get_execution_role()`; from your laptop you pass a role ARN).
- AWS credentials configured locally (`aws configure` / SSO), in a Region where the GPU
  instance is available. `us-east-1` and `us-west-2` are safe choices.
- Python 3.10+ locally with: `pip install sagemaker boto3`.
- Access to the Qwen2.5 model on Hugging Face (public, Apache-2.0 — no token needed for the
  1.5B/7B instruct models).

**Model choice:**

| Model | License | Use when |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | **Default.** Cheapest, fastest, proves the method. |
| `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 | Headline quality; costs a little more time. |

Avoid the 3B and 72B sizes — they use the Qwen research license, not Apache-2.0.
(Source: Qwen2.5 release blog.)

**Folder layout you'll build:**

```
fine-tune-sagemaker/
  GUIDE.md                  (this file)
  data/
    build_dataset.py        you write this (Section 3)
    train.jsonl, val.jsonl, test.jsonl   (generated; samples already here)
  training/
    train.py                you write this (Section 4) - runs on the GPU
    requirements.txt         you write this (Section 4)
    launcher.py             you write this (Section 5) - runs on your laptop
  evaluation/
    scorer.py               you write this (Section 6) - pure metric functions
    evaluate.py             you write this (Section 6) - runs models, writes reports
  cleanup.py                you write this (Section 8)
```

---

## 3. Stage 1 — the dataset

### 3.1 Format

Each line is one training example in the Qwen **chat** format:

```json
{"messages": [
  {"role": "system",    "content": "You are the Horizon Bank virtual assistant..."},
  {"role": "user",      "content": "How much is an international wire?"},
  {"role": "assistant", "content": "International wire transfers cost 35 USD outgoing..."}
]}
```

The trainer renders each example through Qwen's chat template and learns to produce the
assistant turn given the system + user turns.

### 3.2 The four rules that keep the data meaningful (not filler)

1. **Every fact comes from the project's own documents** — `data/loan_products.json` and
   `knowledge-base/*.txt`. You are teaching the model real Horizon Bank policy, not invented text.
2. **Numbers are computed, not typed.** Loan EMI, affordability and DTI are calculated with the
   same formula as the production Lambda (`lambdas/loan_tools/handler.py`), so every answer is
   arithmetically correct and self-consistent.
3. **Each fact appears in several phrasings** so the model learns the fact, not one sentence.
4. **Behaviour is taught too:** loan answers always carry the "indicative, not a final offer"
   line; a salary *range* resolves to its conservative lower bound; under-18 requests are
   declined; out-of-scope questions (crypto tips, "read my balance") are redirected.

### 3.3 The held-out test set is what makes evaluation honest

`test.jsonl` is **disjoint** from training: it uses *different* salary/loan values and *different*
question phrasings. Each test row carries scoring metadata the evaluator uses to produce numbers:

```json
{"messages": [{"role":"system",...},{"role":"user","content":"..."}],
 "category": "fees",
 "required_facts": ["per item"],
 "required_numbers": ["30", "50"],
 "expect_refusal": false,
 "gold": "30 USD per item, up to three per day; not charged if under 50 USD at end of day."}
```

- `category` lets you report accuracy per topic (accounts, fees, loan_eligibility, loan_math,
  scope_refusal).
- `required_facts` / `required_numbers` are the substrings a correct answer must contain.
- `expect_refusal` marks the out-of-scope prompts, where the *right* behaviour is to decline.
- `gold` is a reference answer for your own reading.

### 3.4 The generator (`data/build_dataset.py`)

Write this script. The key helpers are `emi()` (the loan math, mirroring the Lambda), and a
builder for training rows plus a builder for test rows that attaches the scoring metadata.

```python
#!/usr/bin/env python3
"""Build train/val/test JSONL for the Horizon Bank QLoRA fine-tune."""
import json, random
from pathlib import Path

random.seed(7)
ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent

SYSTEM = ("You are the Horizon Bank virtual assistant. Answer questions about Horizon Bank "
          "accounts, fees, cards and loans using only Horizon Bank's published policy. Be "
          "concise, warm and accurate. Use USD. For loan eligibility, always say the result "
          "is indicative and not a final offer, and that final approval needs a full "
          "application, a credit check and income verification. If a question is outside "
          "Horizon Bank banking, politely decline and redirect. Never invent figures.")

PROD = {p["product_type"]: p
        for p in json.loads((ROOT / "data" / "loan_products.json").read_text())}

def emi(principal, annual_rate_pct, months):
    """Reducing-balance EMI, identical to lambdas/loan_tools/handler.py."""
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / months
    f = (1 + r) ** months
    return principal * r * f / (f - 1)

def money(x): return f"{x:,.2f}"

def msg(user, assistant):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}

train = []
def add(u, a): train.append(msg(u, a))

# --- account / fee / policy / scope examples (write ~25-30 of these) ---
add("How much does Everyday Checking cost per month and how do I avoid the fee?",
    "Everyday Checking has an 8 USD monthly maintenance fee. It's waived if you receive at "
    "least 1,500 USD in direct deposits each month, or keep a minimum daily balance of 2,000 USD.")
add("What are the overdraft fees, and when am I not charged?",
    "The overdraft fee is 30 USD per item, up to three fees per day. You're not charged if your "
    "overdraft is under 50 USD at the end of the business day. A returned payment is also 30 USD.")
# ...continue for: student checking (converts at 26), high-yield savings (100 min, six free
# withdrawals), fixed-term deposit (90 days interest penalty), international wire (35/15),
# other-bank ATM (2.50) and abroad (5 + 3%), replacement/express card (free then 5 / 20),
# paper statements (2) and early closure (25), late loan fee (25 or 5% whichever lower),
# early repayment (none for personal/car; home 1% first 3 years), personal vs car vs home loan,
# how eligibility works, salary range -> lower end, under-18 refusal, and 3 scope refusals.

# --- computed loan-eligibility training rows ---
def eligibility(pt, salary, debt, amount, months):
    p = PROD[pt]; name, rate = p["display_name"], p["annual_interest_rate"]
    m = emi(amount, rate, months); dti = (m + debt) / salary
    cap = p["max_salary_multiple"] * salary
    reasons, ok = [], True
    if salary < p["min_monthly_salary"]:
        ok = False; reasons.append(f"the minimum monthly salary for a {name} is {money(p['min_monthly_salary'])} USD")
    if not (p["min_amount"] <= amount <= p["max_amount"]):
        ok = False; reasons.append(f"the amount must be between {money(p['min_amount'])} and {money(p['max_amount'])} USD")
    if amount > cap:
        ok = False; reasons.append(f"the amount is capped at {p['max_salary_multiple']}x your salary ({money(cap)} USD)")
    if dti > p["max_dti_ratio"]:
        ok = False; reasons.append(f"your debt-to-income ratio would be {dti*100:.1f} percent, above the {int(p['max_dti_ratio']*100)} percent maximum")
    q = (f"I'd like a {pt} loan. I earn {money(salary)} a month, already pay {money(debt)} in "
         f"monthly debt, and want {money(amount)} over {months} months.")
    if ok:
        a = (f"Good news - indicatively this looks eligible. A {name} of {money(amount)} USD over "
             f"{months} months at {rate} percent is about {money(m)} USD a month. Your debt-to-income "
             f"ratio is about {dti*100:.1f} percent, within the {int(p['max_dti_ratio']*100)} percent "
             f"limit. This is indicative only, not a final offer - final approval needs a full "
             f"application, a credit check and income verification.")
    else:
        a = (f"Based on what you've told me, this wouldn't pass our indicative check because "
             f"{'; '.join(reasons)}. The repayment on {money(amount)} USD over {months} months at "
             f"{rate} percent would be about {money(m)} USD. This is indicative, not a final decision.")
    return q, a

for s in [("car",6000,500,30000,60), ("personal",3000,400,20000,48), ("personal",2000,0,10000,36),
          ("car",3500,1200,40000,60), ("home",6000,300,250000,300), ("home",4000,0,200000,240),
          ("personal",8000,200,60000,60), ("car",5000,0,5000,24), ("personal",4000,100,15000,36),
          ("home",9000,500,400000,300), ("car",7000,0,60000,72), ("personal",3500,1500,20000,36)]:
    add(*eligibility(*s))

# --- computed EMI-only training rows ---
def emi_qa(pt, amount, months):
    p = PROD[pt]; rate = p["annual_interest_rate"]; m = emi(amount, rate, months); total = m*months
    q = f"What's the monthly payment on a {pt} loan of {money(amount)} over {months} months?"
    a = (f"At the current {p['display_name']} rate of {rate} percent, {money(amount)} USD over "
         f"{months} months is about {money(m)} USD a month, roughly {money(total)} USD total, of "
         f"which about {money(total-amount)} USD is interest. This is indicative.")
    return q, a
for s in [("car",30000,60), ("personal",15000,36), ("home",250000,300),
          ("car",20000,48), ("personal",10000,24), ("home",300000,240)]:
    add(*emi_qa(*s))

random.shuffle(train)
n_val = max(4, int(len(train)*0.1)); val, train = train[:n_val], train[n_val:]

# --- held-out TEST rows with scoring metadata (disjoint values!) ---
def test_row(user, category, facts, numbers, refusal=False, gold=""):
    return {"messages": [{"role":"system","content":SYSTEM},{"role":"user","content":user}],
            "category": category, "required_facts": facts, "required_numbers": numbers,
            "expect_refusal": refusal, "gold": gold}

test = [
  test_row("What's the monthly maintenance fee on Everyday Checking, and how can I get it waived?",
           "accounts", ["waiv"], ["8","1,500","2,000"]),
  test_row("If I'm a student, is there a monthly fee and what happens when I turn 26?",
           "accounts", ["no","convert"], ["26"]),
  test_row("Minimum opening deposit for High-Yield Savings and how many free withdrawals a month?",
           "accounts", ["withdraw"], ["100","six"]),
  test_row("What's the early withdrawal penalty on a fixed-term deposit?", "accounts", ["90 days"], []),
  test_row("How much for an outgoing international wire?", "fees", ["wire"], ["35"]),
  test_row("What's the overdraft fee, and any threshold where I'm not charged?",
           "fees", ["per item"], ["30","50"]),
  test_row("If I miss a loan payment, how is the late fee calculated?",
           "fees", ["whichever is lower"], ["25","5 percent"]),
  test_row("What's the fee for an express replacement debit card?", "fees", ["express"], ["20"]),
]
# loan rows use values NOT in training:
qa = eligibility("car",5500,600,28000,60)
test.append(test_row(qa[0], "loan_eligibility", ["indicative"],
                     [money(emi(28000,PROD["car"]["annual_interest_rate"],60))], gold=qa[1]))
qa = eligibility("personal",2200,0,12000,36)
test.append(test_row(qa[0], "loan_eligibility", ["indicative","2,500"], [], gold=qa[1]))
test.append(test_row("Monthly payment on a 25,000 car loan over 60 months?", "loan_math",
                     ["indicative"], [money(emi(25000,PROD["car"]["annual_interest_rate"],60))]))
test.append(test_row("Should I buy Tesla stock this week?", "scope_refusal",
                     ["can't","Horizon"], [], refusal=True))
test.append(test_row("Can you read out my account balance for me?", "scope_refusal",
                     ["app"], [], refusal=True))

def write(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
write(OUT/"train.jsonl", train); write(OUT/"val.jsonl", val); write(OUT/"test.jsonl", test)
print(f"train {len(train)}  val {len(val)}  test {len(test)}")
```

**Verify:** run `python data/build_dataset.py`. You should see counts printed, and you can sanity-
check the math, e.g. a 30,000 car loan at the 7.9% seed rate over 60 months computes to 606.86/mo —
which matches the production Lambda. Confirm every `required_number` in your test set actually
appears in some training answer, so the evaluation is fair (you're testing what you taught).

---

## 4. Stage 2 — the training entrypoint (`train.py`)

This script runs **inside** the SageMaker GPU container, not on your laptop. SageMaker injects
environment variables telling it where the data is and where to write the model:

| Env var | Meaning |
|---|---|
| `SM_CHANNEL_TRAIN` | directory containing `train.jsonl` (from the "train" input channel) |
| `SM_CHANNEL_VAL` | directory containing `val.jsonl` |
| `SM_MODEL_DIR` | where to save the final model; SageMaker uploads this dir to S3 as `model.tar.gz` |

### 4.1 What the script does, in order

1. Load the tokenizer; set a pad token if missing.
2. Load the base model in **4-bit NF4** (`BitsAndBytesConfig`) — this is the "Q" in QLoRA.
3. `prepare_model_for_kbit_training`, then attach **LoRA** adapters to the attention and MLP
   projection layers — the "LoRA" in QLoRA.
4. Render each example with the Qwen chat template.
5. Train with TRL's `SFTTrainer` for a few epochs.
6. **Merge** the adapters into fp16 and save a complete model to `SM_MODEL_DIR`.
7. Write a small `training_metrics.json` (runtime, sample count) so you have numbers for your report.

### 4.2 `training/train.py`

```python
#!/usr/bin/env python3
"""SageMaker QLoRA training entrypoint (runs inside the Hugging Face DLC)."""
import argparse, json, os

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--train-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "../data"))
    p.add_argument("--val-dir",   default=os.environ.get("SM_CHANNEL_VAL", "../data"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    return p.parse_args()

def load(path):
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]

def main():
    a = args()
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # 4-bit quantized base (the "Q" in QLoRA)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, quantization_config=bnb,
                                                 device_map="auto", dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False

    # LoRA config is handed to the trainer (the SINGLE owner of LoRA). Do NOT also call
    # get_peft_model yourself - doing both detaches the adapters from the trainer's graph
    # and gives grad_norm 0.0 / "element 0 of tensors does not require grad".
    peft_config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

    # Pass raw chat messages; TRL applies the chat template. Do NOT pre-flatten to "text",
    # and do NOT use assistant_only_loss (Qwen2.5's template lacks {% generation %} markers).
    def ds(rows):
        return Dataset.from_dict({"messages": [r["messages"] for r in rows]})
    train_ds = ds(load(os.path.join(a.train_dir, "train.jsonl")))
    val_ds   = ds(load(os.path.join(a.val_dir, "val.jsonl")))

    kw = dict(output_dir="ckpt", per_device_train_batch_size=a.per_device_batch_size,
              gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
              learning_rate=a.lr, warmup_ratio=0.05, logging_steps=1, eval_strategy="epoch",
              save_strategy="no", lr_scheduler_type="cosine", optim="adamw_torch", bf16=True,
              seed=42, max_seq_length=a.max_seq_len, gradient_checkpointing=False, report_to="none")
    try:
        cfg = SFTConfig(**kw)
    except TypeError:                      # drop kwargs an older/newer TRL rejects
        import inspect
        ok = set(inspect.signature(SFTConfig.__init__).parameters)
        cfg = SFTConfig(**{k: v for k, v in kw.items() if k in ok})

    try:                                   # TRL renamed tokenizer -> processing_class
        trainer = SFTTrainer(model=model, processing_class=tok, peft_config=peft_config,
                             train_dataset=train_ds, eval_dataset=val_ds, args=cfg)
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tok, peft_config=peft_config,
                             train_dataset=train_ds, eval_dataset=val_ds, args=cfg)

    # Log how many params actually carry gradients - a non-zero % here is the proof.
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"TRAINABLE PARAMS: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    result = trainer.train()

    # merge adapters -> fp16 full model
    trainer.model.save_pretrained("/tmp/adapters"); tok.save_pretrained("/tmp/adapters")
    del trainer, model; torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float16,
                                                device_map="auto")
    merged = PeftModel.from_pretrained(base, "/tmp/adapters").merge_and_unload()
    os.makedirs(a.model_dir, exist_ok=True)
    merged.save_pretrained(a.model_dir, safe_serialization=True)
    tok.save_pretrained(a.model_dir)
    json.dump({"train_runtime_s": result.metrics.get("train_runtime"),
               "train_loss": result.metrics.get("train_loss"),
               "train_samples": len(train_ds), "model_id": a.model_id},
              open(os.path.join(a.model_dir, "training_metrics.json"), "w"), indent=2)
    print("Saved merged model to", a.model_dir)

if __name__ == "__main__":
    main()
```

Three things in this script were learned the hard way on the container's library stack
(transformers 4.56 / trl 0.23 / peft 0.17 / torch 2.8), and each matters:

- **LoRA is applied once, by the trainer via `peft_config`** — not also by `get_peft_model`.
  Applying it twice detaches the adapters from the graph the trainer optimises, which shows up
  as `grad_norm: 0.0` and then `element 0 of tensors does not require grad`. One owner = gradients
  actually flow.
- **Gradient checkpointing is OFF.** It is only a memory saver, a 1.5B 4-bit model doesn't need it
  on a 24 GB GPU, and it interacts badly with 4-bit + LoRA here.
- **The dataset is raw `messages`** (TRL templates it); we do not pre-flatten to a `text` field and
  do not set `assistant_only_loss` (Qwen2.5's chat template lacks the markers it requires).

The `TRAINABLE PARAMS: ...` print and the `try/except` blocks are there on purpose: the first proves
the adapters are trainable before you pay for a full run, and the second keeps a library version
difference from crashing the job after the GPU has started (which still bills).

### 4.3 `training/requirements.txt`

SageMaker auto-installs this inside the container (it's in `source_dir`). The DLC already ships a
recent `transformers`/`accelerate`, so **do not pin or downgrade those** — only ensure the
PEFT/QLoRA libraries are present:

```
peft>=0.14.0
trl>=0.14.0
bitsandbytes>=0.45.0
datasets>=3.0.0
```

---

## 5. Stage 3 — launching the job (`launcher.py`)

This runs on **your laptop**. It uploads the data, resolves the training container, and starts a
managed GPU job. Three cost guards are built in and you should understand each:

- **`max_run` (hard cap, seconds).** If training ever exceeds it, SageMaker stops the job. This is
  your circuit breaker — the bill cannot run away. Default 3600 (1 hour).
- **`keep_alive_period_in_seconds=0`.** No "warm pool" is kept after the job; nothing lingers.
- **No `.deploy()` call.** We never create an inference endpoint (an endpoint bills hourly until
  deleted). Evaluation is done separately and ephemerally in Stage 4.

### 5.1 Resolving the container image

The `HuggingFace` estimator can take `transformers_version`/`pytorch_version` strings, but those
must match an exact published combo or the job fails at submission. The robust path is to resolve
the image URI with `image_uris.retrieve(...)` (or pass `--image-uri` explicitly). The latest
Hugging Face PyTorch **training** DLC lives under AWS account `763104351884`, e.g.
`763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-pytorch-training:...-gpu-py312-...`.
(Source: Hugging Face "Available DLCs on AWS".)

### 5.2 `training/launcher.py`

```python
#!/usr/bin/env python3
"""Launch the QLoRA fine-tune as a SageMaker training job (runs on your laptop)."""
import argparse

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--instance-type", default="ml.g5.2xlarge",
                   help="ml.g4dn.xlarge (T4, cheapest, for 1.5B) or ml.g5.2xlarge (A10G, for 7B)")
    p.add_argument("--region", default=None)
    p.add_argument("--role", default=None, help="SageMaker execution role ARN")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-run", type=int, default=3600, help="HARD cap (seconds)")
    p.add_argument("--image-uri", default=None)
    p.add_argument("--use-spot", action="store_true", help="~up to 70% cheaper, may wait")
    return p.parse_args()

def main():
    a = args()
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    sess = sagemaker.Session()
    region = a.region or sess.boto_region_name
    role = a.role or sagemaker.get_execution_role()

    prefix = "horizon-bank-qwen25-sft"
    train_s3 = sess.upload_data("data/train.jsonl", key_prefix=f"{prefix}/train")
    val_s3   = sess.upload_data("data/val.jsonl",   key_prefix=f"{prefix}/val")

    image_uri = a.image_uri
    if image_uri is None:
        from sagemaker import image_uris
        image_uri = image_uris.retrieve(framework="huggingface", region=region,
                                        image_scope="training", instance_type=a.instance_type,
                                        base_framework_version="pytorch2.6.0", version="4.49.0")
    print("Training image:", image_uri)

    spot = {}
    if a.use_spot:
        spot = {"use_spot_instances": True, "max_wait": a.max_run + 1800}

    est = HuggingFace(entry_point="train.py", source_dir=".", role=role,
                      instance_type=a.instance_type, instance_count=1, image_uri=image_uri,
                      max_run=a.max_run, keep_alive_period_in_seconds=0, disable_profiler=True,
                      hyperparameters={"model-id": a.model_id, "epochs": a.epochs},
                      base_job_name="hb-qwen25-qlora", **spot)
    est.fit({"train": train_s3, "val": val_s3})

    d = est.latest_training_job.describe()
    print("Model artifact:", est.model_data)
    print("BillableTimeInSeconds:", d.get("BillableTimeInSeconds"))   # <- your real cost input
```

Run it from the `training/` directory (so `source_dir="."` ships `train.py` + `requirements.txt`):

```bash
cd training
python launcher.py --model-id Qwen/Qwen2.5-1.5B-Instruct --instance-type ml.g5.2xlarge
```

**The number that matters:** `BillableTimeInSeconds`. That, times the instance's per-second rate,
is exactly what the job cost. Record it for your report.

Download the trained model afterward:

```bash
aws s3 cp <est.model_data S3 URI> model.tar.gz
mkdir -p merged && tar -xzf model.tar.gz -C merged
```

---

## 6. Stage 4 — the rigorous before/after evaluation

This is the heart of the lab: proving the fine-tune worked, in numbers. The design principle is
**transparent, deterministic metrics on a held-out set** — no black box, no "looks better."

### 6.1 What we measure (and why each)

| Metric | Definition | Why it matters |
|---|---|---|
| **Fact coverage** | % of non-refusal test prompts whose answer contains all its `required_facts` | Did it say the right things? |
| **Exact-number accuracy** | % of required *numbers* present across answers | Fine-tuning's whole point here is correct figures; this catches hallucinated amounts |
| **Refusal correctness** | On `expect_refusal` prompts, did it correctly decline (and on others, did it *not* wrongly refuse)? | A model that refuses everything would score high on safety but be useless; this keeps it honest |
| **Overall pass** | A prompt "passes" if it meets its facts + numbers (or refuses correctly) | One headline number |
| **Per-category** | All of the above, broken down by `category` | Shows *where* fine-tuning helped most |

Run every metric on **both** the base model and the fine-tuned model over the same `test.jsonl`.
The before/after delta is your result.

### 6.2 Keep scoring separate from model-running

Put the pure scoring logic in `evaluation/scorer.py` with **no ML dependencies**, so you can unit-
test it on your laptop in milliseconds. The model-running code goes in `evaluation/evaluate.py`.

### 6.3 `evaluation/scorer.py`

```python
#!/usr/bin/env python3
"""Pure scoring functions. No ML deps - unit-testable on a CPU."""
from __future__ import annotations

def score_one(answer: str, row: dict) -> dict:
    """Score a single model answer against one test row's metadata."""
    low = answer.lower()
    if row.get("expect_refusal"):
        # A correct refusal: mentions it can't help / redirects, and does NOT invent figures.
        refused = any(k in low for k in ["can't", "cannot", "only help", "don't have",
                                         "unable", "app", "online banking"])
        return {"category": row["category"], "passed": refused,
                "facts_ok": refused, "numbers_ok": True, "refusal_ok": refused}
    facts_ok   = all(f.lower() in low for f in row.get("required_facts", []))
    numbers_ok = all(n.lower() in low for n in row.get("required_numbers", []))
    # It should NOT refuse a legitimate question.
    wrongly_refused = any(k in low for k in ["i can't help", "cannot help", "i'm unable"])
    passed = facts_ok and numbers_ok and not wrongly_refused
    return {"category": row["category"], "passed": passed,
            "facts_ok": facts_ok, "numbers_ok": numbers_ok, "refusal_ok": not wrongly_refused}

def aggregate(results: list[dict]) -> dict:
    """Roll individual scores into overall + per-category percentages."""
    def pct(items, key): return round(100 * sum(r[key] for r in items) / len(items), 1) if items else 0.0
    overall = {"n": len(results), "pass_pct": pct(results, "passed"),
               "facts_pct": pct(results, "facts_ok"), "numbers_pct": pct(results, "numbers_ok")}
    by_cat = {}
    for cat in sorted({r["category"] for r in results}):
        items = [r for r in results if r["category"] == cat]
        by_cat[cat] = {"n": len(items), "pass_pct": pct(items, "passed"),
                       "facts_pct": pct(items, "facts_ok"), "numbers_pct": pct(items, "numbers_ok")}
    return {"overall": overall, "by_category": by_cat}
```

You can unit-test this with a few hand-written answers before you ever touch a GPU — e.g. assert a
correct overdraft answer passes and a vague one fails.

> Fairness check: every `required_fact` and `required_number` in `test.jsonl` must be something the
> training data actually teaches. For example `loan_math` rows require the word "indicative"
> because every EMI training answer ends with "This is indicative" — so a correctly tuned model
> learns to say it. Before trusting your numbers, confirm each test requirement appears in at least
> one training answer; otherwise you'd be penalizing the model for something you never taught, and
> even a perfect model would score below 100%.

### 6.4 `evaluation/evaluate.py`

This loads a model, generates a deterministic answer (`do_sample=False`) for each test prompt,
scores it, and writes **three report formats** (JSON for machines, CSV for spreadsheets, Markdown
for humans). Run it once per model, then compare.

```python
#!/usr/bin/env python3
"""Run a model over test.jsonl, score, and write JSON + CSV + Markdown reports."""
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
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                 device_map="auto")
    answers = []
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=220, do_sample=False, pad_token_id=tok.eos_token_id)
        answers.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    return answers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base id (e.g. Qwen/Qwen2.5-1.5B-Instruct) or ./merged")
    ap.add_argument("--label", required=True, help="e.g. base or tuned")
    ap.add_argument("--test", default="../data/test.jsonl")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.test) if l.strip()]
    answers = generate_all(a.model, rows)
    results = [score_one(ans, r) for ans, r in zip(answers, rows)]
    report = aggregate(results)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump({"label": a.label, "report": report,
               "detail": [{"q": next(m["content"] for m in r["messages"] if m["role"]=="user"),
                           "answer": ans, **res}
                          for r, ans, res in zip(rows, answers, results)]},
              open(f"{a.out}/{a.label}.json", "w"), indent=2)
    with open(f"{a.out}/{a.label}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["category","passed","facts_ok","numbers_ok"])
        for res in results: w.writerow([res["category"], res["passed"], res["facts_ok"], res["numbers_ok"]])
    print(f"{a.label}: overall pass {report['overall']['pass_pct']}%  "
          f"numbers {report['overall']['numbers_pct']}%")

if __name__ == "__main__":
    main()
```

### 6.5 Producing the before/after comparison

Run the evaluator twice and write a combined Markdown table:

```bash
cd evaluation
python evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --label base   # the stock model
python evaluate.py --model ../training/merged           --label tuned  # your fine-tune
```

Then a tiny combine step reads `reports/base.json` and `reports/tuned.json` and prints the table
you'll put in your report:

```python
import json
b = json.load(open("reports/base.json"))["report"]
t = json.load(open("reports/tuned.json"))["report"]
print("| Metric | Base | Tuned | Delta |")
print("|---|---|---|---|")
for k in ["pass_pct","facts_pct","numbers_pct"]:
    bo, to = b["overall"][k], t["overall"][k]
    print(f"| {k} | {bo}% | {to}% | +{round(to-bo,1)} |")
```

### 6.6 What you should expect to see

The **base** model typically scores near zero on `numbers_pct` — it has never seen Horizon Bank's
figures, so it confidently states wrong numbers. The **tuned** model should jump to a high pass
rate. The per-category table shows the biggest gains on `fees` and `loan_*` (hard facts), with
`scope_refusal` ideally high on both (good base models already decline politely). That contrast,
in numbers, is the deliverable.

### 6.7 Budget-safe ways to run the evaluation

Do **not** create a persistent SageMaker endpoint for this. Pick one:

- **Cheapest / simplest:** run `evaluate.py` inside the *same* kind of short SageMaker job you used
  for training (reuse the launcher pattern, swap the entry point), or on any GPU box you already
  have, then delete nothing persistent.
- **Or** run it locally if you have an NVIDIA GPU.
- A 1.5B model can even run on CPU for this tiny test set — slow (minutes) but $0.

Whatever you choose, there is no always-on resource, so evaluation adds at most a few more
GPU-minutes to the bill.

---

## 7. Cost table: proving ≤ $5

SageMaker training is billed per second at the instance's on-demand rate. The rates below are
**approximate and region-dependent** — confirm the live figure on the SageMaker pricing page
(Training tab, your Region) before you launch. (Source: AWS SageMaker pricing; "Get compute
capacity for SageMaker Training Jobs".)

| Instance | GPU | ~$/hour (varies by Region) | Good for |
|---|---|---|---|
| `ml.g4dn.xlarge` | T4 16 GB | ~$0.74 – $1.03 | QLoRA on Qwen2.5 1.5B |
| `ml.g5.2xlarge` | A10G 24 GB | ~$1.52 – $2.20 | QLoRA on Qwen2.5 7B |

**The math.** This dataset is tiny, so the GPU work is minutes; budget a generous **45 minutes**
of wall-clock per run (it includes container pull + model download, which you also pay for):

| Scenario | Instance | Rate used | Time | Cost |
|---|---|---|---|---|
| 1.5B training | g4dn.xlarge | $1.00/hr | 0.75 hr | **~$0.75** |
| 7B training | g5.2xlarge | $2.20/hr (worst case) | 0.75 hr | **~$1.65** |
| Evaluation (both models) | same GPU, short | — | ~0.3 hr | **~$0.30–$0.70** |
| S3 storage of data + model | — | cents | — | **~$0.05** |

Even the 7B path with evaluation lands around **$2–$3**, comfortably under $5. Guards that keep it
there: the `max_run` circuit breaker, no warm pool, and never deploying an endpoint. For extra
margin, add `--use-spot` to the launcher (often up to ~70% cheaper).

> The exact dollar figure depends on your Region's live rate and the actual
> `BillableTimeInSeconds` the job reports. The guide gives you the formula and where to read both;
> plug your real numbers in for a precise total.

---

## 8. Mandatory cleanup (`cleanup.py`)

Per-second training jobs stop billing when they finish, so the main thing to clean up is **stored
artifacts**. Do this when you're done:

1. **Delete the model + data from S3.** Find the bucket/prefix the launcher used (the default
   SageMaker bucket, prefix `horizon-bank-qwen25-sft`, plus the training job's output path).

   ```bash
   aws s3 rm s3://<sagemaker-default-bucket>/horizon-bank-qwen25-sft/ --recursive
   aws s3 rm s3://<sagemaker-default-bucket>/<training-job-output-prefix>/ --recursive
   ```

2. **Confirm no endpoints exist** (there shouldn't be any, since we never deployed):

   ```bash
   aws sagemaker list-endpoints
   aws sagemaker list-endpoint-configs
   ```
   If anything is listed from experimentation, delete it — endpoints bill hourly:
   ```bash
   aws sagemaker delete-endpoint --endpoint-name <name>
   aws sagemaker delete-endpoint-config --endpoint-config-name <name>
   ```

3. **Delete local artifacts** (`model.tar.gz`, `merged/`) if you don't need them.

4. Training jobs themselves cannot be "deleted" and cost nothing once stopped — no action needed.

A short `cleanup.py` using `boto3` can do steps 1–2 in one run; follow the same resource names the
launcher used.

---

## 9. Sources

- QLoRA: Dettmers et al., [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Hugging Face PEFT quantization: [peft quantization guide](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md)
- Qwen2.5 sizes and license: [Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5/)
- Bedrock native fine-tuning only supports a fixed model list (not Qwen): [Customize a model with fine-tuning in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html)
- SageMaker per-second billing, no commitment: [Amazon SageMaker pricing](https://aws.amazon.com/sagemaker/pricing/)
- Reading per-Region training instance rates: [Get compute capacity for SageMaker Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/train-get-capacity.html)
- Hugging Face SageMaker training & DLCs: [HF on SageMaker training](https://huggingface.co/docs/sagemaker/train), [Available DLCs on AWS](https://huggingface.co/docs/sagemaker/main/dlcs/available)
- `max_run` / estimator parameters: [SageMaker Python SDK Estimators](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)
- Managed Spot training: [Managed Spot Training in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)

Content from these sources was rephrased for compliance with licensing restrictions.
