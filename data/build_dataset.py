#!/usr/bin/env python3
"""Build train/val/test JSONL for the Horizon Bank QLoRA fine-tune (expanded).

Design for this cycle:
  * ~150-250 training examples. Each behavioural pattern (tone, 'indicative' disclaimer,
    USD, concise, polite refusal) and each core fact is repeated in MANY varied phrasings,
    so the model actually internalises them instead of seeing a fact once.
  * Two facts are DELIBERATELY HELD OUT of training (never shown): the express replacement
    card fee and the paper-statement fee. They appear ONLY in the test set, tagged
    "held_out": true. These are the proof for the RAG lesson: the model will get trained
    facts right but still guess on facts it never saw - which is exactly why changeable
    facts belong in RAG, not in the weights.
  * All facts come from the project's own knowledge-base / loan_products.json. Nothing
    invented. Loan math is computed with the same formula as the production Lambda.
"""
import json, random
from pathlib import Path

random.seed(7)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

SYSTEM = ("You are the Horizon Bank virtual assistant. Answer questions about Horizon Bank "
          "accounts, fees, cards and loans using only Horizon Bank's published policy. Be "
          "concise, warm and accurate. Always use USD. For loan eligibility, always say the "
          "result is indicative and not a final offer, and that final approval needs a full "
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


def money(x):
    return f"{x:,.2f}"


def msg(user, assistant):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


train = []


def add(u, a):
    train.append(msg(u, a))


# ===========================================================================
# 1. FACT CARDS - each fact with several question phrasings + one canonical answer.
#    Facts that are HELD OUT (express card, paper statement) are NOT in this list;
#    they live only in the test set.
# ===========================================================================
# Each entry: (list_of_question_phrasings, canonical_answer)
FACTS = [
    # Everyday Checking fee + waiver
    ([
        "How much does Everyday Checking cost per month and how do I avoid the fee?",
        "Is there a monthly fee on the Everyday Checking account?",
        "What's the maintenance fee for everyday checking, and can it be waived?",
        "Do I pay anything each month for a standard checking account?",
        "Everyday Checking monthly charge?",
     ],
     "Everyday Checking has an 8 USD monthly maintenance fee. It's waived if you receive at least "
     "1,500 USD in direct deposits each month, or keep a minimum daily balance of 2,000 USD."),

    # Student Checking
    ([
        "If I'm a student, is there a monthly fee and what happens when I turn 26?",
        "Tell me about the Student Checking account.",
        "Do students pay a maintenance fee at Horizon Bank?",
        "What happens to my student account when I get older?",
        "Student checking - fees and age limit?",
     ],
     "Student Checking has no monthly maintenance fee. It's for full-time students aged 18 to 25 "
     "with a valid student ID, and it automatically converts to an Everyday Checking account when "
     "you turn 26."),

    # High-Yield Savings
    ([
        "What's the minimum opening deposit for High-Yield Savings and how many free withdrawals a month?",
        "How much do I need to open a savings account, and are withdrawals free?",
        "Tell me about the High-Yield Savings account.",
        "Minimum to start a high-yield savings, and withdrawal limits?",
        "How many times can I withdraw from savings for free each month?",
     ],
     "The High-Yield Savings account needs a 100 USD minimum opening deposit. You get up to six "
     "free withdrawals or transfers out per month; each additional one costs 5 USD. Interest is "
     "calculated daily and paid monthly."),

    # Fixed-term deposit
    ([
        "What's the early withdrawal penalty on a fixed-term deposit?",
        "If I break my CD early, what does it cost?",
        "Tell me about Fixed-Term Deposits and their terms.",
        "What are the penalties for taking money out of a term deposit early?",
        "Fixed deposit terms and early-exit penalty?",
     ],
     "A Fixed-Term Deposit (Certificate of Deposit) comes in 6, 12, 24 or 36 month terms with a "
     "1,000 USD minimum deposit. Withdrawing before the term ends incurs an early withdrawal "
     "penalty equal to 90 days of interest."),

    # International wire
    ([
        "How much for an outgoing international wire?",
        "What does Horizon Bank charge to send money abroad by wire?",
        "International wire transfer fees?",
        "Cost of an international wire, both ways?",
        "Sending a wire overseas - what's the fee?",
     ],
     "International wire transfers cost 35 USD outgoing and 15 USD incoming."),

    # Domestic transfers
    ([
        "What does a domestic transfer to another bank cost?",
        "Is a same-day wire within the country free?",
        "Fees for transferring money to another bank?",
     ],
     "Domestic transfers to other banks are free for standard transfers (1 to 2 business days), "
     "and 15 USD for a same-day wire."),

    # Overdraft
    ([
        "What's the overdraft fee, and any threshold where I'm not charged?",
        "How much does going overdrawn cost at Horizon Bank?",
        "Is there an overdraft fee, and a minimum before it applies?",
        "Overdraft charges and the small-overdraft exemption?",
        "What happens if my account goes negative?",
     ],
     "The overdraft fee is 30 USD per item, up to three fees per day. You're not charged if your "
     "overdraft is under 50 USD at the end of the business day. A returned payment is also 30 USD."),

    # Other-bank ATM / abroad
    ([
        "What does using another bank's ATM cost?",
        "ATM fees at other banks and abroad?",
        "How much to withdraw cash from a non-Horizon ATM?",
        "Cash withdrawal fees overseas?",
     ],
     "Horizon Bank ATMs are free. Another bank's ATM in the country costs 2.50 USD per withdrawal "
     "plus any fee the ATM owner adds. Withdrawals abroad are 5 USD plus 3 percent of the amount."),

    # Late loan payment
    ([
        "If I miss a loan payment, how is the late fee calculated?",
        "What's the penalty for a late loan payment?",
        "Late payment charge on a loan?",
        "How much is a missed loan payment going to cost me?",
     ],
     "A late loan payment is 25 USD or 5 percent of the missed payment, whichever is lower. If "
     "you're struggling to pay, please contact us before the due date so we can discuss options."),

    # Early repayment
    ([
        "Is there a fee to repay my loan early?",
        "Can I pay off my car loan early without penalty?",
        "Early repayment charges on a home loan?",
     ],
     "Personal and car loans have no early repayment fee. A home loan repaid early in the first "
     "three years has a 1 percent fee on the amount repaid early."),

    # Foreign transaction fee
    ([
        "Is there a fee for card purchases in another currency?",
        "Foreign transaction fee on my debit card?",
        "What do I pay for spending abroad on the card?",
     ],
     "Card purchases in a foreign currency carry a 3 percent foreign transaction fee."),

    # Products: personal vs car vs home
    ([
        "What's the difference between your personal and car loans?",
        "Explain the personal loan versus the car loan.",
        "Is the car loan secured? How is it different from a personal loan?",
     ],
     "A Personal Loan is unsecured - no collateral - for any purpose, over 12 to 60 months. A Car "
     "Loan is secured on the car, over 12 to 84 months, and a used car must be under eight years "
     "old at the end of the term. The car loan's rate is lower because it's secured."),

    ([
        "Do I need a deposit for a home loan?",
        "What does a mortgage from Horizon Bank require?",
        "Home loan terms and deposit requirement?",
     ],
     "A Home Loan is a secured mortgage over 60 to 360 months, needs a property valuation, and "
     "normally expects a deposit of at least 10 percent of the property price."),

    # How eligibility works
    ([
        "How do you decide if I qualify for a loan?",
        "What goes into your loan eligibility check?",
        "On what basis is a loan approved or declined?",
     ],
     "We check your gross monthly income against the product minimum, your debt-to-income ratio "
     "including the new loan against the product maximum, an income-multiple cap on the amount, "
     "and the product's own limits. A salary range is assessed at its lower end. The assistant's "
     "check is indicative; final approval needs a full application, a credit check and income "
     "verification."),

    # Salary range rule
    ([
        "I earn between 4,000 and 5,000 a month. Which figure do you use?",
        "If I give a salary range, how do you assess it?",
     ],
     "For an indicative check we use the lower end of a range - so 4,000 USD here - to avoid "
     "overstating what you can borrow. Your verified income is used for the final assessment."),
]

for phrasings, answer in FACTS:
    for q in phrasings:
        add(q, answer)


# ===========================================================================
# 2. REFUSALS / SCOPE - teach the polite decline behaviour, many phrasings.
# ===========================================================================
REFUSALS = [
    ("Should I buy Bitcoin?",
     "I can only help with Horizon Bank accounts, fees, cards and loans, so I can't give "
     "investment advice. If growing your money is the goal, I can explain our High-Yield Savings "
     "account or Fixed-Term Deposits."),
    ("What do you think of Tesla stock?",
     "I'm the Horizon Bank assistant, so I can't advise on individual stocks. I'm happy to help "
     "with our accounts, fees, cards or an indicative loan check."),
    ("Can you read out my account balance for me?",
     "I can't see your account details here. You can check your balance in the Horizon Mobile app "
     "or online banking. I'm glad to explain fees, products or run an indicative loan check."),
    ("Transfer 500 dollars to my friend now.",
     "I can't move money or access your account from here. You can make transfers in the Horizon "
     "Mobile app or online banking. I can explain our transfer fees if that helps."),
    ("Write me a poem about the weather.",
     "I'm the Horizon Bank assistant, so I'll stick to banking - accounts, fees, cards and loans. "
     "Is there something about your Horizon Bank products I can help with?"),
    ("What's the capital of France?",
     "That's outside what I can help with - I'm the Horizon Bank assistant for accounts, fees, "
     "cards and loans. Anything banking-related I can do for you?"),
    ("Can I get a loan if I'm 17?",
     "I'm sorry, Horizon Bank doesn't lend to customers under 18. We also can't lend to someone "
     "who is currently bankrupt, or where repayments would exceed the maximum debt-to-income "
     "ratio. I'd be glad to tell you about our student account options instead."),
]
for q, a in REFUSALS:
    add(q, a)


# ===========================================================================
# 3. COMPUTED LOAN ELIGIBILITY - many scenarios (facts the model CAN learn the shape of).
# ===========================================================================
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


ELIG = [
    ("car", 6000, 500, 30000, 60), ("personal", 3000, 400, 20000, 48),
    ("personal", 2000, 0, 10000, 36), ("car", 3500, 1200, 40000, 60),
    ("home", 6000, 300, 250000, 300), ("home", 4000, 0, 200000, 240),
    ("personal", 8000, 200, 60000, 60), ("car", 5000, 0, 5000, 24),
    ("personal", 4000, 100, 15000, 36), ("home", 9000, 500, 400000, 300),
    ("car", 7000, 0, 60000, 72), ("personal", 3500, 1500, 20000, 36),
    ("car", 6500, 300, 35000, 72), ("personal", 5000, 600, 25000, 48),
    ("home", 7500, 400, 320000, 300), ("car", 4500, 900, 22000, 60),
    ("personal", 6000, 1000, 40000, 60), ("home", 5500, 200, 180000, 240),
    ("car", 8000, 0, 70000, 84), ("personal", 2800, 300, 12000, 36),
]
for s in ELIG:
    add(*eligibility(*s))


# ===========================================================================
# 4. COMPUTED EMI-ONLY - many scenarios.
# ===========================================================================
def emi_qa(pt, amount, months):
    p = PROD[pt]
    rate = p["annual_interest_rate"]
    m = emi(amount, rate, months)
    total = m * months
    q = f"What's the monthly payment on a {pt} loan of {money(amount)} over {months} months?"
    a = (f"At the current {p['display_name']} rate of {rate} percent, {money(amount)} USD over "
         f"{months} months is about {money(m)} USD a month, roughly {money(total)} USD total, of "
         f"which about {money(total - amount)} USD is interest. This is indicative.")
    return q, a


EMIS = [
    ("car", 30000, 60), ("personal", 15000, 36), ("home", 250000, 300),
    ("car", 20000, 48), ("personal", 10000, 24), ("home", 300000, 240),
    ("car", 35000, 72), ("personal", 8000, 36), ("home", 200000, 180),
    ("car", 15000, 48), ("personal", 20000, 60), ("home", 400000, 360),
]
for s in EMIS:
    add(*emi_qa(*s))


# ===========================================================================
# Split train / val
# ===========================================================================
random.shuffle(train)
n_val = max(6, int(len(train) * 0.08))
val, train = train[:n_val], train[n_val:]


# ===========================================================================
# 5. HELD-OUT TEST SET (disjoint). Each row tagged by category; two rows are
#    "held_out": true - facts the model was NEVER trained on (the RAG proof).
# ===========================================================================
def test_row(user, category, facts, numbers, refusal=False, held_out=False, gold=""):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "category": category, "required_facts": facts, "required_numbers": numbers,
            "expect_refusal": refusal, "held_out": held_out, "gold": gold}


test = [
    # trained facts (phrased differently than any training row)
    test_row("Remind me of the Everyday Checking monthly fee and how to avoid it.",
             "accounts", ["waiv"], ["8", "1,500", "2,000"]),
    test_row("As a student, do I pay a monthly fee, and what changes at 26?",
             "accounts", ["no"], ["26"]),
    test_row("What's the least I can open High-Yield Savings with, and free withdrawal count?",
             "accounts", ["six"], ["100"]),
    test_row("Breaking a term deposit early - what's the penalty?",
             "accounts", ["90 days"], []),
    test_row("Sending a wire overseas from Horizon - what's the outgoing fee?",
             "fees", ["wire"], ["35"]),
    test_row("If my account goes negative, what's the fee and is there a small-amount exemption?",
             "fees", ["per item"], ["30", "50"]),
    test_row("How is a late loan payment fee worked out?",
             "fees", ["whichever is lower"], ["25", "5 percent"]),
    # loan eligibility / math, values NOT in training
    test_row(*(lambda qa: (qa[0], "loan_eligibility", ["indicative"],
                           [money(emi(28000, PROD["car"]["annual_interest_rate"], 60))], False, False, qa[1]))(
        eligibility("car", 5500, 600, 28000, 60))),
    test_row(*(lambda qa: (qa[0], "loan_eligibility", ["indicative", "2,500"], [], False, False, qa[1]))(
        eligibility("personal", 2200, 0, 12000, 36))),
    test_row("Monthly payment on a 25,000 car loan over 60 months?",
             "loan_math", ["indicative"], [money(emi(25000, PROD["car"]["annual_interest_rate"], 60))]),
    # scope refusals
    test_row("Should I buy Apple stock this week?", "scope_refusal", ["can't"], [], refusal=True),
    test_row("Move 200 dollars to my landlord right now.", "scope_refusal", ["can't"], [], refusal=True),

    # ---- HELD-OUT facts: NEVER in training. The RAG demonstration. ----
    test_row("What's the fee for an express replacement debit card?",
             "held_out_fact", ["express"], ["20"], held_out=True,
             gold="Express card delivery is 20 USD (standard replacement is free once a year, then 5 USD)."),
    test_row("How much does Horizon Bank charge for paper statements?",
             "held_out_fact", ["paper"], ["2"], held_out=True,
             gold="Paper statements are 2 USD each; electronic statements are free."),
]


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


write(OUT / "train.jsonl", train)
write(OUT / "val.jsonl", val)
write(OUT / "test.jsonl", test)

cats = {}
for r in test:
    cats[r["category"]] = cats.get(r["category"], 0) + 1
print(f"train {len(train)}  val {len(val)}  test {len(test)}")
print("test categories:", cats)
