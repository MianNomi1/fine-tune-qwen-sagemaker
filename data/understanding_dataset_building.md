# Build-along: write `build_dataset.py` yourself (for someone new to Python)

This document teaches you to **write** `build_dataset.py` from scratch. Read it top to bottom; at
each step you type a small block into your own file and run a check. By the end you'll have the
complete, working script — and you'll understand every line because you wrote it.

No prior Python assumed.

- **Parts A–C** explain *why* the file exists, whether you need to install anything, and the five
  Python ideas you'll use.
- **Part D** is the build-along: Steps 1–10, each "type this → why → check it."
- **Parts E–F** are a self-check and what to do next.

> Create an empty file first: `fine-tune-sagemaker/data/build_dataset.py`. You'll grow it step by
> step. Keep a terminal open in the `fine-tune-sagemaker` folder to run the checks.

---

## Part A — Why does this file exist at all?

### A.1 The big picture

You are teaching an AI model to answer Horizon Bank questions. Teaching an AI = showing it
**many examples** of "here is a question, here is the ideal answer." The model studies those
examples and learns the pattern. Those examples have to come from somewhere. **This file is the
factory that produces them.**

```
Horizon Bank's facts   ->   build_dataset.py   ->   three files of training examples
(data/loan_products.json)    (this factory)          (train.jsonl, val.jsonl, test.jsonl)
```

Without it you'd hand-type hundreds of Q&A pairs, getting every number right and keeping them
consistent. The script does that correctly and repeatably.

### A.2 What it produces

Three files, same *kind* of thing (lists of example conversations), different jobs:

| File | Job | Analogy |
|---|---|---|
| `train.jsonl` | Examples the model **learns from** | The textbook you study |
| `val.jsonl` | A few examples checked **during** training to watch progress | Practice quiz |
| `test.jsonl` | Examples kept **hidden** until the end, to grade the model honestly | Final exam |

Golden rule: test examples must never appear in training. The script enforces this by giving test
questions *different numbers* than training questions.

### A.3 What one example looks like

```
SYSTEM:    "You are the Horizon Bank virtual assistant. Be accurate, add disclaimers..."
USER:      "How much is an international wire?"
ASSISTANT: "International wire transfers cost 35 USD outgoing and 15 USD incoming..."
```

The model learns: given this SYSTEM instruction and this kind of USER question, produce this kind
of ASSISTANT answer.

### A.4 Why compute numbers instead of typing them

Some answers contain math (loan payments, debt ratios). If you typed "$606.86" by hand, one typo
teaches the model a wrong fact it will repeat forever. Instead the script **calculates** every
number with the same formula the real bank uses. The computer can't fat-finger arithmetic.

---

## Part B — Do I need to install anything?

**No.** This file uses only Python's **standard library** (ships inside Python). It imports:

```python
import json      # read/write JSON - built in
import random    # shuffle examples - built in
from pathlib import Path   # file paths - built in
```

No `pip install` needed. You just need Python 3.10+.

> Check: in a terminal run `python3 --version`. If it prints `Python 3.1x.x`, you're ready. macOS
> usually already has it.

The heavy AI packages (`torch`, `transformers`) belong to a *different* file (`train.py`) that runs
on a cloud GPU. **This** file is plain Python that runs instantly on your laptop. That split is
deliberate: data prep is cheap and local; training is expensive and remote.

---

## Part C — Five Python ideas you'll use (30 seconds each)

1. **Variable** — a name for a value. `x = 5`.
2. **String** — text in quotes: `"hello"`. Strings written next to each other inside parentheses
   are glued together:
   ```python
   s = ("hello "
        "world")      # -> "hello world"
   ```
3. **List** — ordered collection in `[]`: `[1, 2, 3]`. You can append to it.
4. **Dictionary** — key→value table in `{}`: `{"rate": 7.9}`. Read with `d["rate"]`.
5. **Function** — a reusable mini-machine, defined with `def`, takes inputs, `return`s an output:
   ```python
   def double(n):
       return n * 2
   double(5)          # -> 10
   ```

That's all you need. Everything below combines these.

---

## Part D — Build-along: write the script in 10 steps

Type each step into `data/build_dataset.py` in order. Run the check after each one.

### Step 1 — Imports and a reproducible shuffle

Type at the top of the file:

```python
import json, random
from pathlib import Path

random.seed(7)
```

**Why.** We load the standard-library tools. `random.seed(7)` makes the later shuffle *repeatable*:
without it every run reorders examples differently; with it, everyone gets the identical dataset.
The `7` is an arbitrary fixed starting point.

**Check.** Run `python3 data/build_dataset.py`. Nothing should print, and there should be **no
error**. A silent, clean run means your imports are valid. (We'll add output later.)

---

### Step 2 — Figure out where the files are

Add below Step 1:

```python
ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent
```

**Why.** `__file__` is "the path to this script." `.resolve()` makes it absolute. `.parent` is the
folder the script is in (`data/`) — where we'll **write** output (`OUT`). `.parents[2]` goes up two
folders to the project root, where `data/loan_products.json` lives (`ROOT`). Computing paths this
way means the script works on any computer, not just yours.

**Check.** Temporarily add `print(ROOT)` and `print(OUT)` at the bottom, run the file, and confirm
`OUT` ends in `/fine-tune-sagemaker/data` and `ROOT` ends in your project root folder. Then delete
those two `print`s.

---

### Step 3 — The system prompt

Add:

```python
SYSTEM = ("You are the Horizon Bank virtual assistant. Answer questions about Horizon Bank "
          "accounts, fees, cards and loans using only Horizon Bank's published policy. Be "
          "concise, warm and accurate. Use USD. For loan eligibility, always say the result "
          "is indicative and not a final offer, and that final approval needs a full "
          "application, a credit check and income verification. If a question is outside "
          "Horizon Bank banking, politely decline and redirect. Never invent figures.")
```

**Why.** This one string becomes the SYSTEM turn in **every** example — the assistant's rules. The
model learns the link between this instruction and good answers, so you must use the *same* system
prompt later when you ask real questions.

**Check.** Add `print(len(SYSTEM))` at the bottom, run, confirm it prints a number (a few hundred).
Remove the print.

---

### Step 4 — Load the real loan facts

Add:

```python
PROD = {p["product_type"]: p
        for p in json.loads((ROOT / "data" / "loan_products.json").read_text())}
```

**Why.** `(ROOT / "data" / "loan_products.json")` builds the path (`/` joins folders with `Path`).
`.read_text()` reads the file into a string. `json.loads(...)` turns that JSON string into a Python
**list** of loan dicts. The `{ ... for p in ... }` is a **dict comprehension** that reshapes the
list into a lookup table keyed by product type:

```python
# file is:   [ {"product_type":"car", "annual_interest_rate":7.9, ...}, {...}, {...} ]
# PROD is:   { "car": {...}, "personal": {...}, "home": {...} }
```

So later `PROD["car"]["annual_interest_rate"]` instantly gives `7.9` — no searching.

**Check.** Add `print(PROD.keys())` then `print(PROD["car"]["annual_interest_rate"])`, run, and
confirm you see the three product names and the car rate. Remove the prints. (If you get a
`FileNotFoundError`, your `ROOT` is wrong — revisit Step 2.)

---

### Step 5 — The loan math function

Add:

```python
def emi(principal, annual_rate_pct, months):
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / months
    f = (1 + r) ** months
    return principal * r * f / (f - 1)

def money(x):
    return f"{x:,.2f}"
```

**Why.** `emi` computes the fixed monthly loan payment (Equated Monthly Instalment). It converts a
yearly percent to a monthly fraction (`/100/12`), guards against divide-by-zero, then applies the
standard bank formula (`**` is "to the power of"). This is the **same formula the production system
uses**, so dataset numbers match reality. `money` formats a number as `"1,234.56"` (`:,.2f` =
commas + two decimals) so all amounts look consistent.

**Check.** Add `print(money(emi(30000, 7.9, 60)))`, run, and confirm it prints `606.86`. That's the
known-correct payment for a 30,000 car loan at 7.9% over 60 months. Remove the print.

---

### Step 6 — Turn a Q&A into one example, and collect them

Add:

```python
def msg(user, assistant):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}

train = []
def add(u, a):
    train.append(msg(u, a))
```

**Why.** `msg` wraps a question and its ideal answer into the exact dict shape the trainer wants
(the three turns from A.3). `train = []` is an empty list to collect examples. `add` builds an
example and **appends** it to that list, so calling `add("q", "a")` grows `train` by one.

**Check.** After this block add a quick test:

```python
add("test question", "test answer")
print(len(train))        # should print 1
train.clear()            # reset so the test example isn't saved
```

Run, confirm it prints `1`, then delete those three test lines.

---

### Step 7 — Write the hand-authored examples

These are human-written answers (facts, not math), so you type them as `add(...)` calls. Write
**~25–30** covering the facts below. Here are three to start; continue in the same style.

```python
add("How much does Everyday Checking cost per month and how do I avoid the fee?",
    "Everyday Checking has an 8 USD monthly maintenance fee. It's waived if you receive at least "
    "1,500 USD in direct deposits each month, or keep a 2,000 USD minimum daily balance.")

add("What are the overdraft fees, and when am I not charged?",
    "The overdraft fee is 30 USD per item, up to three fees per day. You're not charged if your "
    "overdraft is under 50 USD at the end of the business day. A returned payment is also 30 USD.")

add("Should I buy Bitcoin?",
    "I can only help with Horizon Bank accounts, fees, cards and loans, so I can't give investment "
    "advice. If growing your money is the goal, I can explain our High-Yield Savings account or "
    "Fixed-Term Deposits.")
```

**Facts to cover** (pull the exact numbers from `knowledge-base/06-fees-and-charges.txt` and
`knowledge-base/03-accounts-and-products.txt`):

- Everyday Checking fee + waiver; Student Checking (no fee, converts at 26); High-Yield Savings
  (100 USD minimum, six free withdrawals then 5 USD); Fixed-Term Deposit (6/12/24/36 months, 90
  days interest penalty).
- Fees: international wire (35 out / 15 in), other-bank ATM (2.50) and abroad (5 + 3%), replacement
  card (free once/year then 5) and express (20), paper statements (2), early closure (25),
  overdraft (30, none under 50), late loan (25 or 5% whichever **lower**), early repayment (none
  for personal/car; home 1% in first 3 years).
- Policy: personal vs car vs home loan; how eligibility works; salary range → use lower end;
  under-18 refusal; two more scope refusals.

**Why multiple phrasings.** Add a couple of differently-worded questions for the same fact (e.g.
"Is there a monthly fee?" and "What does Everyday Checking cost?"). The model should learn the
*fact*, not one sentence.

**Check.** Add `print(len(train))` at the bottom, run, and confirm it climbs as you add more.
Keep this print for now — it's handy.

---

### Step 8 — Generate loan examples from scenarios (the computed ones)

Add the eligibility builder:

```python
def eligibility(pt, salary, debt, amount, months):
    p = PROD[pt]
    name, rate = p["display_name"], p["annual_interest_rate"]
    m = emi(amount, rate, months)
    dti = (m + debt) / salary
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
```

**Why, slowly.**
- Inputs: product type, salary, existing debt, loan amount, months.
- `m = emi(...)` computes the monthly payment; `dti` is the debt-to-income ratio; `cap` is the
  income-multiple limit. These are the real eligibility rules from the bank's policy doc.
- `reasons, ok = [], True` sets up an empty reasons list and an "eligible so far?" flag.
- Each `if` checks one rule; if it fails, we flip `ok` to `False` and record a reason. (`a; b` on
  one line just runs two statements — a compact style.)
- `f"...{value}..."` is an **f-string**: it drops a variable's value into the text. `{dti*100:.1f}`
  means "dti as a percent, one decimal."
- `'; '.join(reasons)` joins the reason strings with "; " between them.
- It `return`s a `(question, answer)` pair.

Now generate many at once:

```python
for s in [("car",6000,500,30000,60), ("personal",3000,400,20000,48), ("personal",2000,0,10000,36),
          ("car",3500,1200,40000,60), ("home",6000,300,250000,300), ("home",4000,0,200000,240),
          ("personal",8000,200,60000,60), ("car",5000,0,5000,24), ("personal",4000,100,15000,36),
          ("home",9000,500,400000,300), ("car",7000,0,60000,72), ("personal",3500,1500,20000,36)]:
    add(*eligibility(*s))
```

**Why the `*`.** Each item `s` is a **tuple** (a fixed group in parentheses). `eligibility(*s)`
**unpacks** it so `("car",6000,500,30000,60)` becomes `eligibility("car",6000,500,30000,60)`.
`eligibility` returns a `(q, a)` pair, and `add(*...)` unpacks that pair into `add(q, a)`. Result:
12 fully-correct examples from 12 short tuples, with all math computed.

Add a second builder for plain "what's the payment?" questions:

```python
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
```

**Check.** Run the file. Your `print(len(train))` should now show roughly 40+ (your hand-written
ones + 12 + 6). If you get a `KeyError`, a product type is misspelled; if a math error, re-check
Step 5.

---

### Step 9 — Split into train and validation

Add:

```python
random.shuffle(train)
n_val = max(4, int(len(train) * 0.1))
val, train = train[:n_val], train[n_val:]
```

**Why.** `shuffle` mixes topics so training isn't all-fees-then-all-loans (that can bias learning);
it's repeatable because of the seed in Step 1. `int(len(train) * 0.1)` is 10% of the examples;
`max(4, ...)` keeps at least 4. `train[:n_val]` is "the first n_val items" and `train[n_val:]` is
"the rest" — this is **list slicing**. We take the first slice as `val` and keep the rest as
`train`. They don't overlap.

**Check.** Add `print(len(train), len(val))`, run, confirm two numbers that add up to your earlier
total, with `val` around 4. Keep this print.

---

### Step 10 — Build the held-out test set, then write all three files

First the test-row helper and the test list:

```python
def test_row(user, category, facts, numbers, refusal=False, gold=""):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "category": category, "required_facts": facts,
            "required_numbers": numbers, "expect_refusal": refusal, "gold": gold}

test = [
    test_row("What's the monthly maintenance fee on Everyday Checking, and how can I get it waived?",
             "accounts", ["waiv"], ["8", "1,500", "2,000"]),
    test_row("If I'm a student, is there a monthly fee and what happens when I turn 26?",
             "accounts", ["no", "convert"], ["26"]),
    test_row("Minimum opening deposit for High-Yield Savings and how many free withdrawals a month?",
             "accounts", ["withdraw"], ["100", "six"]),
    test_row("What's the early withdrawal penalty on a fixed-term deposit?", "accounts", ["90 days"], []),
    test_row("How much for an outgoing international wire?", "fees", ["wire"], ["35"]),
    test_row("What's the overdraft fee, and any threshold where I'm not charged?",
             "fees", ["per item"], ["30", "50"]),
    test_row("If I miss a loan payment, how is the late fee calculated?",
             "fees", ["whichever is lower"], ["25", "5 percent"]),
    test_row("What's the fee for an express replacement debit card?", "fees", ["express"], ["20"]),
    test_row("Should I buy Tesla stock this week?", "scope_refusal", ["can't", "Horizon"], [], refusal=True),
    test_row("Can you read out my account balance for me?", "scope_refusal", ["app"], [], refusal=True),
]

# loan test rows use numbers NOT seen in training, so the exam is honest:
qa = eligibility("car", 5500, 600, 28000, 60)
test.append(test_row(qa[0], "loan_eligibility", ["indicative"],
                     [money(emi(28000, PROD["car"]["annual_interest_rate"], 60))], gold=qa[1]))
qa = eligibility("personal", 2200, 0, 12000, 36)
test.append(test_row(qa[0], "loan_eligibility", ["indicative", "2,500"], [], gold=qa[1]))
test.append(test_row("Monthly payment on a 25,000 car loan over 60 months?", "loan_math",
                     ["indicative"], [money(emi(25000, PROD["car"]["annual_interest_rate"], 60))]))
```

**Why.** A test row has **no assistant turn** — the model must produce the answer at exam time.
Instead it carries grading metadata: `category` (for per-topic scores), `required_facts` /
`required_numbers` (what a correct answer must contain, so grading is automatic), `expect_refusal`
(True for off-topic questions), and `gold` (a reference answer for you). `refusal=False` and
`gold=""` are **default values** — you can omit them. The loan rows deliberately use amounts/salaries
not in Step 8, so the model can't just recite.

> Fairness rule: every `required_fact`/`required_number` in a test row must be something your
> training answers actually teach. Example: the `loan_math` row requires "indicative" because every
> `emi_qa` answer ends with "This is indicative." Otherwise you'd be grading the model on something
> you never taught, and even a perfect model couldn't score 100%.

Now the writer and the final calls:

```python
def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

write(OUT / "train.jsonl", train)
write(OUT / "val.jsonl", val)
write(OUT / "test.jsonl", test)
print(f"train {len(train)}  val {len(val)}  test {len(test)}")
```

**Why.** `write` saves a list to a file. `with open(path, "w") as f:` opens it for **w**riting and
auto-closes it afterward. For each example, `json.dumps(r, ensure_ascii=False)` turns the dict into
a JSON string (the opposite of `json.loads`), and `+ "\n"` puts each on its own line — that's what
makes it JSON *Lines*. The three calls save the three lists; the `print` reports the counts.

**Check (final).** Run `python3 data/build_dataset.py`. You should see a line like
`train 37  val 4  test 15`, and three new files in `data/`. Open `data/train.jsonl` and read a
couple of lines — each is one example with the three turns. You've built the whole script.

---

## Part E — Prove it to yourself

Run these quick checks after the final build:

1. **Count the lines** of each file: `wc -l data/*.jsonl`. The numbers should match the printed
   counts.
2. **Confirm the math** is correct, e.g. a car loan of 30,000 over 60 months should contain
   `606.86` somewhere in training:
   ```bash
   grep -c "606.86" data/train.jsonl
   ```
   A number ≥ 1 means your computed answers landed in the data.
3. **Confirm train and test don't overlap**: the test loan uses salary 5,500 / amount 28,000, which
   should **not** appear in `train.jsonl`:
   ```bash
   grep -c "28,000" data/train.jsonl        # expect 0
   ```

If all three pass, your dataset is correct, consistent, and honestly split.

---

## Part F — Self-check questions

<details>
<summary>1. Why compute loan numbers with <code>emi()</code> instead of typing them?</summary>
So every number is correct and consistent — no typo for the model to learn and repeat.
</details>
<details>
<summary>2. Why must test questions use different numbers than training?</summary>
So the final exam is honest; the model must generalize, not recite memorized answers.
</details>
<details>
<summary>3. Do you need to <code>pip install</code> anything for this file?</summary>
No — only Python's standard library (<code>json</code>, <code>random</code>, <code>pathlib</code>).
</details>
<details>
<summary>4. What are the three turns in a training example, and how many does a test row have?</summary>
Training: system, user, assistant. Test: only system and user (the model must produce the answer).
</details>
<details>
<summary>5. What does <code>random.seed(7)</code> buy you?</summary>
A repeatable shuffle — everyone who runs the script gets the identical dataset.
</details>

---

## Part G — What's next

Once this file makes sense and runs cleanly, the next script to understand is `train.py` — the one
that actually fine-tunes the model on a cloud GPU. It's more involved and uses the heavy packages.
If this build-along style worked for you, ask me and I'll write the same step-by-step companion for
`train.py`, the launcher, and the evaluation scripts.
