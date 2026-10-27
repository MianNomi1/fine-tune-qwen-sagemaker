#!/usr/bin/env python3
"""SageMaker QLoRA training entrypoint (runs inside the Hugging Face DLC).

Key design decisions (learned the hard way on the transformers 4.56 / trl 0.23 /
peft 0.17 / torch 2.8 stack in the training container):

  * LoRA is applied by SFTTrainer via `peft_config`, NOT by calling get_peft_model
    ourselves. Doing both caused the adapters to be detached from the graph the
    trainer optimises, which showed up as grad_norm: 0.0 / "element 0 of tensors does
    not require grad". One owner of LoRA = gradients actually flow.
  * Gradient checkpointing is OFF. A 1.5B model in 4-bit on a 24 GB A10G has plenty of
    memory, and checkpointing interacts badly with 4-bit + LoRA on this stack.
  * The dataset is passed as raw chat `messages`; TRL applies the chat template.
    We do NOT pre-flatten to a "text" field and we do NOT use assistant_only_loss
    (Qwen2.5's template lacks the {% generation %} markers it needs).
"""
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
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


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

    # ---- 4-bit quantized base (the "Q" in QLoRA) ----
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

    # ---- LoRA config: handed to the trainer, which is the SINGLE owner of LoRA ----
    peft_config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ---- dataset: raw chat messages; TRL applies the chat template itself ----
    def ds(rows):
        return Dataset.from_dict({"messages": [r["messages"] for r in rows]})

    train_ds = ds(load(os.path.join(a.train_dir, "train.jsonl")))
    val_ds = ds(load(os.path.join(a.val_dir, "val.jsonl")))

    # ---- training config (built defensively against TRL version differences) ----
    kw = dict(
        output_dir="ckpt",
        per_device_train_batch_size=a.per_device_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        warmup_ratio=0.05,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        lr_scheduler_type="cosine",
        optim="adamw_torch",          # plain optimiser: avoids 8-bit optimiser edge cases
        bf16=True,
        seed=42,
        max_seq_length=a.max_seq_len,
        gradient_checkpointing=False,
        report_to="none",
    )
    try:
        cfg = SFTConfig(**kw)
    except TypeError:
        import inspect
        allowed = set(inspect.signature(SFTConfig.__init__).parameters)
        cfg = SFTConfig(**{k: v for k, v in kw.items() if k in allowed})

    # ---- trainer applies LoRA via peft_config (one owner) ----
    def make_trainer(**extra):
        try:
            return SFTTrainer(model=model, processing_class=tok, peft_config=peft_config,
                              train_dataset=train_ds, eval_dataset=val_ds, args=cfg, **extra)
        except TypeError:
            return SFTTrainer(model=model, tokenizer=tok, peft_config=peft_config,
                              train_dataset=train_ds, eval_dataset=val_ds, args=cfg, **extra)

    trainer = make_trainer()

    # Sanity check in the logs: how many params actually carry gradients.
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"TRAINABLE PARAMS: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    result = trainer.train()

    # ---- merge adapters into fp16 and save a complete model to SM_MODEL_DIR ----
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
               "train_loss": result.metrics.get("train_loss"),
               "train_samples": len(train_ds), "model_id": a.model_id},
              open(os.path.join(a.model_dir, "training_metrics.json"), "w"), indent=2)
    print("Saved merged model to", a.model_dir)


if __name__ == "__main__":
    main()
