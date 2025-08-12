import argparse
import subprocess
import sys


def finetune(dataset: str, base: str, output: str, epochs: int = 3, batch_size: int = 8, lr: float = 5e-5):
    cmd = [
        "ollama", "finetune", base,
        "--dataset", dataset,
        "--output", output,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(lr),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("New model:", output)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fine_tune_data.jsonl")
    p.add_argument("--base", default="deepseek-r1:8b")
    p.add_argument("--output", default="deepseek-finetuned")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()
    finetune(args.dataset, args.base, args.output, args.epochs, args.batch_size, args.lr)

