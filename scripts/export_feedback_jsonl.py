import json
import argparse
from app.personalization.models import create_session, Feedback


def export_jsonl(out_path: str = "fine_tune_data.jsonl", min_rating: int = 1, max_rating: int = 2):
    Session = create_session()
    s = Session()
    q = s.query(Feedback).filter(Feedback.rating <= max_rating)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in q:
            # Prefer user-provided corrections in comments; fallback to answer
            target = (rec.comments or rec.answer or "").strip()
            if not target:
                continue
            prompt = (rec.query or "").strip() + "\n\n### Response:"
            entry = {"prompt": prompt, "completion": " " + target + "  \n"}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    s.close()
    print(f"Exported {count} examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fine_tune_data.jsonl")
    parser.add_argument("--max_rating", type=int, default=2)
    args = parser.parse_args()
    export_jsonl(args.out, max_rating=args.max_rating)

