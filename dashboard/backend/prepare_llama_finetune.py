from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.train_notebook_artifact import load_csv_data


def _seed_examples(rows: list[dict]) -> list[dict[str, str]]:
    by_store: dict[int, list[dict]] = {}
    for r in rows:
        by_store.setdefault(int(r["Store"]), []).append(r)
    for k in by_store:
        by_store[k] = sorted(by_store[k], key=lambda x: str(x["Date"]))

    all_sales = [float(r["Weekly_Sales"]) for r in rows]
    peak = max(rows, key=lambda r: float(r["Weekly_Sales"]))
    low = min(rows, key=lambda r: float(r["Weekly_Sales"]))
    avg_sales = mean(all_sales) if all_sales else 0.0
    top_store = max(((sid, mean(float(x["Weekly_Sales"]) for x in vals)) for sid, vals in by_store.items()), key=lambda x: x[1])[0]

    return [
        {
            "instruction": "Who made taAI?",
            "response": "taAI was built by Panshulaj Pechetty and Rishabh Gupta.",
        },
        {
            "instruction": "What is the average weekly sales across the dataset?",
            "response": f"Average weekly sales is approximately ${avg_sales:,.0f}.",
        },
        {
            "instruction": "When was the highest sales week?",
            "response": f"Highest weekly sales occurred on {peak['Date']} with ${float(peak['Weekly_Sales']):,.0f}.",
        },
        {
            "instruction": "When was the lowest sales week?",
            "response": f"Lowest weekly sales occurred on {low['Date']} with ${float(low['Weekly_Sales']):,.0f}.",
        },
        {
            "instruction": "Which store performs best on average?",
            "response": f"Store {top_store} has the highest average weekly sales in this dataset.",
        },
    ]


def build_jsonl(output: Path) -> None:
    rows = load_csv_data()
    examples = _seed_examples(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for ex in examples:
            msg = {
                "messages": [
                    {"role": "system", "content": "You are taAI, a Walmart economist assistant."},
                    {"role": "user", "content": ex["instruction"]},
                    {"role": "assistant", "content": ex["response"]},
                ]
            }
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small llama fine-tune chat dataset for taAI.")
    parser.add_argument(
        "--output",
        default="backend/finetune/taai_llama_finetune.jsonl",
        help="Output JSONL path relative to dashboard/",
    )
    args = parser.parse_args()
    output_path = ROOT / args.output
    build_jsonl(output_path)


if __name__ == "__main__":
    main()
