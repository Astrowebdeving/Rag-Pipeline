"""Feedback Improvement Loop Orchestrator.

Commands:
  - start: run periodic export -> finetune -> reload
  - once: run one cycle immediately
  - stop: (no-op placeholder; run under a process manager to stop)

Usage:
  source ../ragvenv/bin/activate && python scripts/feedback_loop.py once \
    --base deepseek-r1:8b --out deepseek-finetuned --interval 86400
"""

import argparse
import os
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def export_feedback(out_path: str) -> None:
    cmd = [
        "python", os.path.join(PROJECT_ROOT, "scripts", "export_feedback_jsonl.py"),
        "--out", out_path,
    ]
    subprocess.run(cmd, check=True)


def finetune(dataset: str, base: str, output: str) -> None:
    cmd = [
        "python", os.path.join(PROJECT_ROOT, "scripts", "run_ollama_finetune.py"),
        "--dataset", dataset,
        "--base", base,
        "--output", output,
    ]
    subprocess.run(cmd, check=True)


def reload_model(model_name: str, api_key: str | None = None) -> None:
    import requests
    url = "http://localhost:5000/api/model/reload"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = requests.post(url, json={"model_name": model_name}, headers=headers, timeout=10)
    resp.raise_for_status()


def model_exists(name: str) -> bool:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        return any(line.split()[0] == name for line in out.strip().splitlines() if line.strip())
    except Exception:
        return False


def run_cycle(base: str, output: str, api_key: str | None) -> None:
    dataset = os.path.join(PROJECT_ROOT, "fine_tune_data.jsonl")
    export_feedback(dataset)
    # If an existing finetuned model already exists, continuously refine it by using it as the new base
    effective_base = output if model_exists(output) else base
    finetune(dataset, effective_base, output)
    reload_model(output, api_key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["once", "start"], help="Run one cycle or loop")
    p.add_argument("--base", default="deepseek-r1:8b")
    p.add_argument("--out", default="deepseek-finetuned")
    p.add_argument("--interval", type=int, default=86400, help="Seconds between cycles when running start")
    args = p.parse_args()

    api_key = os.environ.get("RAG_API_KEY")

    if args.command == "once":
        run_cycle(args.base, args.out, api_key)
        print("✅ Feedback loop cycle completed")
        return

    if args.command == "start":
        print("▶️ Starting feedback loop. Press Ctrl+C to stop.")
        try:
            while True:
                run_cycle(args.base, args.out, api_key)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("⏹️ Feedback loop stopped by user")


if __name__ == "__main__":
    main()

