import json
import pandas as pd
import os
from tqdm import tqdm

# -------------------------
# Load JSON or JSONL
# -------------------------
def load_file(path):
    rows = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except:
                    pass
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
        rows = data
    return rows

# -------------------------
# Evaluate ToCAD formatted results
# -------------------------
def evaluate_tocad_file(file_path):
    data = load_file(file_path)
    rows = []

    for item in data:
        expected = str(item.get("expected", "")).strip()
        pred = str(item.get("final_output", "")).strip()

        correct = (expected.lower() == pred.lower())

        rows.append({
            "file": os.path.basename(file_path),
            "word": item.get("word", ""),
            "a": item.get("a", ""),
            "b": item.get("b", ""),
            "expected": expected,
            "predicted": pred,
            "correct": correct,
            "type": "swap"  # these files are swap-only
        })

    if not rows:
        return None, None

    df = pd.DataFrame(rows)

    summary = {
        "file": os.path.basename(file_path),
        "samples": len(df),
        "overall_accuracy": round(df["correct"].mean() * 100, 2),
        "swap_accuracy": round(df["correct"].mean() * 100, 2)  # same, only swap
    }

    return df, summary


# ======================================================
# MAIN EXECUTION
# ======================================================

tocad_files = [
    "llama3_tocad_swap_eval.json",
    "llama3_tocad_swap_eval.jsonl"
]

all_dfs = []
summaries = []

for f in tocad_files:
    if os.path.exists(f):
        df, summary = evaluate_tocad_file(f)
        if df is not None:
            all_dfs.append(df)
            summaries.append(summary)
    else:
        print(f"⚠ Missing file: {f}")

# Output results
if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv("tocad_eval_details_correct.csv", index=False)
    pd.DataFrame(summaries).to_csv("tocad_eval_summary_correct.csv", index=False)

    print("\n📊 Results generated:")
    print("→ tocad_eval_details_correct.csv")
    print("→ tocad_eval_summary_correct.csv\n")
else:
    print("\n❌ No valid ToCAD files evaluated!")
