# Build-along: write `train.py` yourself (for someone new to Python and ML)

This teaches you to **write** `train.py` — the script that actually fine-tunes the model. Read top
to bottom; type each block into your own file and reason through the check.

**Read this first — it changes how you use this guide:**

> `train.py` does **not** run on your Mac. It runs on a rented **GPU** computer inside Amazon
> SageMaker. The packages it uses (`torch`, `transformers`, `bitsandbytes`) don't even install on
> macOS, and the work needs a GPU your Mac doesn't have. So our "checks" here are **syntax checks
> and reasoning**, not "run it and see output." You'll run the real thing in the cloud later, via
> the launcher (a separate file).

- **Parts A–C:** why this file exists, the ML concepts (GPU, quantization, LoRA, QLoRA, epochs),
  and the one Python syntax idea you need.
- **Part D:** build-along, Steps 1–9.
- **Parts E–F:** how to syntax-check, and what's next.

> Create/empty the file: `fine-tune-sagemaker/training/train.py`. You already have
> `training/requirements.txt`.

---

## Part A — Why does this file exist?

In the previous file (`build_dataset.py`) you created the **examples**. This file is the **teacher**
that sits a fresh Qwen2.5 model down with those examples and trains it until it answers like the
Horizon Bank assistant.

```
train.jsonl (your examples)  +  Qwen2.5 base model  ->  train.py  ->  a fine-tuned model
                                                          (the teacher)   (saved to disk)
```

The script runs on a GPU machine that SageMaker starts for you. SageMaker hands it three things via
**environment variables** (settings the surrounding system passes in):

| Variable | Meaning |
|---|---|
| `SM_CHANNEL_TRAIN` | folder where `train.jsonl` was placed |
| `SM_CHANNEL_VAL` | folder where `val.jsonl` was placed |
| `SM_MODEL_DIR` | folder to save the finished model into (SageMaker uploads it to cloud storage) |

So the shape of the script is: read those locations → load model + data → train → save to
`SM_MODEL_DIR`.

---

## Part B — The ML concepts you need (plain language)

You don't need math. You need these ideas.

### B.1 What "the model" is
A large language model is billions of numbers called **weights**. "Fine-tuning" means nudging some
of those numbers so the model behaves the way you want on your data.

### B.2 GPU
A **GPU** is a chip that does the massive parallel arithmetic training needs. CPUs (what your Mac
uses for normal work) are far too slow for this. That's why training runs on a cloud GPU.

### B.3 Quantization (the "Q" in QLoRA)
The model's weights are normally stored as high-precision numbers (16 bits each), which takes a lot
of GPU memory. **Quantization** stores them in a compressed, lower-precision form — here **4 bits**
each ("NF4" is the specific 4-bit format). Same model, far less memory. This is what lets a big
model fit on a modest, cheap GPU.

### B.4 LoRA (the "LoRA" in QLoRA)
Instead of nudging *all* billions of weights (expensive, memory-hungry), **LoRA** freezes the
original weights and adds a few small, trainable "adapter" matrices on the side. You train only
those tiny adapters. You get most of the benefit at a fraction of the cost. LoRA belongs to a
family called **PEFT** (Parameter-Efficient Fine-Tuning).

### B.5 QLoRA = Q + LoRA
Load the model **quantized to 4-bit** (B.3), freeze it, and **train small LoRA adapters** on top
(B.4). That's QLoRA. It's the whole reason we can fine-tune Qwen2.5 cheaply.

### B.6 Merge
At the end, we fold the trained adapters back into the base weights to produce **one ordinary
model** you can use anywhere — no special adapter loading needed later.

### B.7 A few training words you'll see
- **Tokenizer** — turns text into numbers the model reads, and back. Each model ships its own.
- **Chat template** — the exact formatting that marks "this is the system turn, this is the user
  turn," etc. The tokenizer knows its model's template.
- **Epoch** — one full pass over all training examples. We do a few (e.g. 3) so the model sees each
  example several times.
- **Batch size / gradient accumulation** — how many examples the GPU processes at once. Small
  numbers use less memory; "accumulation" simulates a bigger batch without more memory.
- **`SFTTrainer`** — a ready-made trainer (from the `trl` library) for this kind of supervised
  fine-tuning. It handles the training loop so you don't write it by hand.

---

## Part C — The one Python idea you need here

**`try` / `except`** — "attempt something; if it fails a certain way, do a fallback instead."

```python
try:
    risky_thing()
except TypeError:        # only catches this specific kind of error
    fallback_thing()
```

Why it matters in this file: the training libraries change their function names between versions,
and the exact version inside the cloud container isn't fully under our control. `try/except` lets
the script work with either the old or the new name — so a tiny version difference doesn't crash a
job **after** the GPU has started costing money. That resilience is a real cost protection, not
decoration.

Everything else is the variables/strings/lists/dicts/functions you already met.

---

## Part D — Build-along: write `train.py` in 9 steps

Type each block in order.

### Step 1 — Imports that are always safe, and the argument reader

At the very top:

```python
import argparse, json, os
```

**Why.** These three are standard library (safe to import anywhere, even on your Mac): `argparse`
reads command-line options, `json` reads your data files, `os` reads environment variables.

Notice we do **not** import `torch`/`transformers` at the top. We'll import those *inside* the main
function (Step 3 onward), because they only exist on the GPU container, not on your Mac. Importing
them at the top would make the file impossible to even syntax-check locally.

Now the options reader:

```python
def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--epochs", type=float, default=3.0)
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
```

**Why.** `argparse` lets SageMaker (and you) pass settings like `--epochs 3`. Each `add_argument`
declares one option with a **default**, so the script runs even if nothing is passed. The important
ones:
- `--model-id` which Qwen2.5 to fine-tune (1.5B by default).
- the data/model directories default to the SageMaker environment variables via
  `os.environ.get("NAME", fallback)` — "use the env var if set, else the fallback." The fallback
  (`../data`, `model`) lets you reason about it locally; on SageMaker the env vars win.
- `type=float` / `type=int` convert the text argument into a number.

**Check.** This block is pure standard library. Run `python3 -c "import ast; ast.parse(open('training/train.py').read()); print('syntax ok')"` — it should say `syntax ok`. (This parses the file
without running it, so no GPU packages are needed.)

---

### Step 2 — A tiny helper to read JSONL

```python
def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
```

**Why.** Your dataset files are JSONL (one JSON object per line). This opens a file and, for each
non-empty line, parses it into a Python dict — returning a **list of dicts**. `l.strip()` skips
blank lines. This is the mirror image of the `write` function you wrote in `build_dataset.py`.

**Check.** Still standard library — re-run the syntax check from Step 1.

---

### Step 3 — Start `main()` and load the tokenizer

```python
def main():
    a = args()
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
```

(Note: we import `LoraConfig` and `PeftModel` but **not** `get_peft_model` — we never wrap the model
ourselves; the trainer does it. See Step 5.)

**Why.** `a = args()` reads the options. Then we import the heavy libraries **inside** `main` (see
Step 1 note) — they load on the GPU container. `AutoTokenizer.from_pretrained(a.model_id)` downloads
the tokenizer that matches your chosen model. The `pad_token` lines set a "padding" token if the
model doesn't define one (padding makes examples the same length during training; harmless
housekeeping).

**Check.** The file won't *run* on your Mac now (those imports will fail locally), but the **syntax
check** from Step 1 still works and should still say `syntax ok`. That's the right check at this
stage — you're confirming you typed valid Python, not executing it.

---

### Step 4 — Load the base model in 4-bit (the "Q")

```python
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model_id, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False
```

**Why.** `BitsAndBytesConfig` is the recipe for the 4-bit quantization from Part B.3: `load_in_4bit`
turns it on, `nf4` picks the 4-bit format, `double_quant` squeezes a little more memory,
`compute_dtype` is the precision used during the math. `from_pretrained(..., quantization_config=bnb)`
downloads Qwen2.5 and loads it **compressed**. `device_map="auto"` places it on the GPU.
`prepare_model_for_kbit_training(..., use_gradient_checkpointing=False)` makes the quantized model
trainable without turning on checkpointing (which we deliberately avoid — see Step 7).
`model.config.use_cache = False` silences a harmless warning during training.

This is the step that lets a big model fit on a cheap GPU.

**Check.** Syntax check. Also reason it through: this is the "Q" of QLoRA — confirm you can point to
*where the model becomes 4-bit* (the `bnb` config passed into `from_pretrained`).

---

### Step 5 — Define the LoRA config (the "LoRA")

```python
    peft_config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
```

**Why.** `LoraConfig` describes the adapters: `r` and `lora_alpha` control their size/strength (16
is a sane default), `task_type="CAUSAL_LM"` says "this is a text-generation model,"
`target_modules` lists the internal layers to attach adapters to (these names are Qwen2.5's
attention and feed-forward projections — standard for this model family).

**Important — one owner of LoRA.** Notice we do NOT call `get_peft_model(model, ...)` here. We only
*describe* the adapters and hand this `peft_config` to the trainer in Step 8; the trainer attaches
them. Early drafts of this lab applied LoRA twice (once with `get_peft_model` and again implicitly in
the trainer), and the adapters ended up detached from the graph the trainer actually optimises. The
symptom was brutal and silent: `grad_norm: 0.0` every step and a loss that never moved, because
there was literally nothing for gradients to flow into. The rule to remember: **apply LoRA in
exactly one place.** We choose the trainer.

**Check.** Syntax check. Reason it through: this is the "LoRA" of QLoRA, but it's only the *recipe* —
the adapters get attached in Step 8.

---

### Step 6 — Load your examples as chat messages

```python
    def ds(rows):
        return Dataset.from_dict({"messages": [r["messages"] for r in rows]})

    train_ds = ds(load(os.path.join(a.train_dir, "train.jsonl")))
    val_ds   = ds(load(os.path.join(a.val_dir, "val.jsonl")))
```

**Why.** Each example is a `messages` list (system/user/assistant). Hand those **raw messages** to
the dataset and let TRL apply the chat template itself — modern TRL understands the conversational
format natively.

**Important — do NOT pre-flatten to a `text` field.** An earlier version of this lab did
`tok.apply_chat_template(...)` here and stored a `"text"` string. On the container's TRL version that
quietly broke the loss masking (the trainer couldn't tell which tokens were the assistant's answer),
so every token got masked and the model learned nothing — another route to the frozen-loss /
`grad_norm: 0.0` symptom. Passing structured `messages` and letting TRL template them is the reliable
path. `os.path.join` builds the path to each file inside the folder SageMaker gave us.

**Check.** Syntax check. Reason it through: input is your JSONL rows; output is a `Dataset` whose
single column is `messages` — the conversational format TRL expects.

---

### Step 7 — Configure the trainer (resilient to library versions)

```python
    kw = dict(output_dir="ckpt", per_device_train_batch_size=a.per_device_batch_size,
              gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
              learning_rate=a.lr, warmup_ratio=0.05, logging_steps=1, eval_strategy="epoch",
              save_strategy="no", lr_scheduler_type="cosine", optim="adamw_torch", bf16=True,
              seed=42, max_seq_length=a.max_seq_len, gradient_checkpointing=False, report_to="none")
    try:
        cfg = SFTConfig(**kw)
    except TypeError:
        import inspect
        ok = set(inspect.signature(SFTConfig.__init__).parameters)
        cfg = SFTConfig(**{k: v for k, v in kw.items() if k in ok})
```

**Why.** `kw` is a dictionary of training settings:
- `num_train_epochs` how many passes over the data (Part B.7); `learning_rate` how big each nudge
  is; `per_device_train_batch_size` + `gradient_accumulation_steps` how many examples per step;
  `eval_strategy="epoch"` check the validation set each epoch; `save_strategy="no"` we save
  ourselves at the end; `report_to="none"` don't phone home to logging services.
- `logging_steps=1` so you see `grad_norm` and `loss` on **every** step — essential for spotting a
  broken run early instead of after it finishes.
- `gradient_checkpointing=False` — checkpointing is only a memory saver, and on this 4-bit + LoRA
  stack it breaks the gradient path. A 1.5B model in 4-bit has ample memory on a 24 GB GPU, so we
  simply don't use it.
- `optim="adamw_torch"` — the plain optimiser, to avoid 8-bit-optimiser edge cases.
- `SFTConfig(**kw)` passes all those settings in. The `**` **unpacks a dict into named arguments**
  (like the `*` unpacked a tuple in the dataset guide, but for key=value pairs).
- The `try/except` (Part C) protects against version drift: if this `trl` version doesn't accept one
  of the settings, we use `inspect` to ask the config which arguments it *does* accept, and pass
  only those. So the job won't crash on a slightly different library version.

**Check.** Syntax check. Reason it through: `**kw` means "take every key in `kw` and pass it as
`key=value`." The `except` branch is "if that was rejected, filter down to supported keys and retry."

---

### Step 8 — Create the trainer (it attaches LoRA) and train

```python
    try:
        trainer = SFTTrainer(model=model, processing_class=tok, peft_config=peft_config,
                             train_dataset=train_ds, eval_dataset=val_ds, args=cfg)
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tok, peft_config=peft_config,
                             train_dataset=train_ds, eval_dataset=val_ds, args=cfg)

    # Prove, before training, that something is actually trainable.
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"TRAINABLE PARAMS: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    result = trainer.train()
```

**Why.** `SFTTrainer` is the ready-made training loop (Part B.7). It needs the model, the tokenizer,
the train/val datasets, and the config. The critical argument is **`peft_config=peft_config`** — this
is where the LoRA adapters from Step 5 actually get attached, by the trainer and nowhere else. That
single-owner rule is what makes gradients flow (see the hard-won note in Step 5).

The `try/except` handles a real naming change: newer `trl` calls the tokenizer argument
`processing_class`, older calls it `tokenizer`. We try the new name; if that version rejects it
(`TypeError`), we fall back to the old name.

The `TRAINABLE PARAMS` print is a cheap insurance policy: it should show roughly 1% trainable. If it
ever shows `0.0000%`, stop — the adapters aren't attached and training would be pointless.
`trainer.train()` then runs the actual training.

**Check.** Syntax check. Reason it through: `peft_config` is the one place LoRA is applied; the two
branches differ only in the tokenizer argument name.

---

### Step 9 — Merge adapters, save the finished model, record metrics

```python
    trainer.model.save_pretrained("/tmp/adapters")
    tok.save_pretrained("/tmp/adapters")
    del trainer, model
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        a.model_id, dtype=torch.float16, device_map="auto")
    merged = PeftModel.from_pretrained(base, "/tmp/adapters").merge_and_unload()
    os.makedirs(a.model_dir, exist_ok=True)
    merged.save_pretrained(a.model_dir, safe_serialization=True)
    tok.save_pretrained(a.model_dir)

    json.dump({"train_runtime_s": result.metrics.get("train_runtime"),
               "train_samples": len(train_ds), "model_id": a.model_id},
              open(os.path.join(a.model_dir, "training_metrics.json"), "w"), indent=2)
    print("Saved merged model to", a.model_dir)

if __name__ == "__main__":
    main()
```

**Why.** After training, the valuable output is the LoRA adapters. We:
1. Save the adapters to a temp folder, and free GPU memory (`del ...` + `empty_cache()`).
2. Reload the base model in normal 16-bit precision, attach the adapters with
   `PeftModel.from_pretrained`, and `merge_and_unload()` — this **folds the adapters into the
   weights** (Part B.6), giving one ordinary model.
3. `merged.save_pretrained(a.model_dir, safe_serialization=True)` writes that model (plus the
   tokenizer) into `SM_MODEL_DIR`. SageMaker automatically uploads whatever is in that folder to
   cloud storage — that's how you get your model back.
4. `json.dump(...)` writes a small metrics file (how long training took, how many samples) so you
   have real numbers for your report.

The last two lines, `if __name__ == "__main__": main()`, mean "if this file is run as a program,
call `main()`." Standard Python entry point.

**Check.** Final syntax check:
```bash
python3 -c "import ast; ast.parse(open('training/train.py').read()); print('syntax ok')"
```
If it says `syntax ok`, your script is well-formed. It won't *run* here — that's expected; it runs
on the GPU via the launcher, the next file.

---

## Part E — How to check a file you can't run locally

You can't execute `train.py` on your Mac (no GPU, packages don't install). So your local checks are:

1. **Syntax check** (does it parse as valid Python?):
   ```bash
   python3 -c "import ast; ast.parse(open('training/train.py').read()); print('syntax ok')"
   ```
2. **Read-through check** — point to each QLoRA piece in your own code:
   - Where does the model become 4-bit? (Step 4, the `bnb` config)
   - Where are the LoRA adapters attached? (Step 8, the trainer's `peft_config` — defined in Step 5)
   - Where are adapters merged into one model? (Step 9, `merge_and_unload`)
   - Where is the model saved for SageMaker to collect? (Step 9, `save_pretrained(a.model_dir)`)
3. **requirements check** — open `training/requirements.txt` and confirm it lists `peft`, `trl`,
   `bitsandbytes`, `datasets` (lower bounds, not pinned), and does **not** downgrade `transformers`.
   SageMaker installs these inside the container at job start.

The real end-to-end run happens when you launch the SageMaker job (next file). There you'll watch
the logs print `print_trainable_parameters` output and finally "Saved merged model to ...".

---

## Part F — Self-check and what's next

<details>
<summary>1. Why are torch/transformers imported inside <code>main()</code> instead of at the top?</summary>
So the file can be syntax-checked on your Mac, where those GPU-only packages aren't installed.
</details>
<details>
<summary>2. In QLoRA, which step is the "Q" and which is the "LoRA"?</summary>
Step 4 (load the model in 4-bit) is the Q; Step 5 (attach LoRA adapters) is the LoRA.
</details>
<details>
<summary>3. Why the two <code>try/except</code> blocks?</summary>
Training libraries rename things between versions; the fallbacks keep the job from crashing on a
different library version — crashing after the GPU started would waste money.
</details>
<details>
<summary>4. What does <code>merge_and_unload()</code> give you?</summary>
One ordinary model with the adapters folded in, so you don't need special adapter-loading later.
</details>
<details>
<summary>5. How does your model get out of the GPU container and back to you?</summary>
By saving it into <code>SM_MODEL_DIR</code>; SageMaker uploads that folder to cloud storage
automatically.
</details>

**Next:** the **launcher** (`training/launcher.py`) — the small script that *runs on your laptop* to
start the SageMaker GPU job, with the hard cost cap. After that, the **evaluation** scripts that
produce your before/after numbers. Ask me and I'll write the same build-along for either.
