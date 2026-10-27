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
    p.add_argument("--epochs", type=float, default=10.0)
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
    train_s3 = sess.upload_data("../data/train.jsonl", key_prefix=f"{prefix}/train")
    val_s3   = sess.upload_data("../data/val.jsonl",   key_prefix=f"{prefix}/val")

    image_uri = a.image_uri
    if image_uri is None:
        from sagemaker import image_uris
        image_uri = image_uris.retrieve(framework="huggingface", region=region,
                                        image_scope="training", instance_type=a.instance_type,
                                        base_framework_version="pytorch2.8.0", version="4.56.2")
    print("Training image:", image_uri)

    spot = {}
    if a.use_spot:
        spot = {"use_spot_instances": True, "max_wait": a.max_run + 1800}

    est = HuggingFace(entry_point="train.py", source_dir=".", role=role,
                      instance_type=a.instance_type, instance_count=1, image_uri=image_uri,
                      py_version="py312",
                      max_run=a.max_run, keep_alive_period_in_seconds=0, disable_profiler=True,
                      hyperparameters={"model-id": a.model_id, "epochs": a.epochs},
                      base_job_name="hb-qwen25-qlora", **spot)
    est.fit({"train": train_s3, "val": val_s3})

    d = est.latest_training_job.describe()
    print("Model artifact:", est.model_data)
    print("BillableTimeInSeconds:", d.get("BillableTimeInSeconds"))   # <- your real cost input


if __name__ == "__main__":
    main()
