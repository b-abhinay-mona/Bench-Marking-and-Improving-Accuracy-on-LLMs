import os
import json
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------
# SAFE edit type detector
# -------------------------------------------------------
def get_edit_type(question):
    if not isinstance(question, str):
        return "other"
    q = question.lower()
    if "substitute" in q:
        return "substitute"
    if "remove" in q:
        return "remove"
    if "add" in q:
        return "add"
    if "swap" in q:
        return "swap"
    return "other"

# -------------------------------------------------------
# Read all .json files
# -------------------------------------------------------
folder = "."
json_files = [f for f in os.listdir(folder) if f.endswith(".json")]

print(f"\n📁 Found {len(json_files)} JSON files:")
for f in json_files:
    print(" -", f)

all_dfs = []          # merged all rows
file_summaries = []   # per-file accuracy table

# -------------------------------------------------------
# Process each json file SEPARATELY
# -------------------------------------------------------
for file in tqdm(json_files, desc="Processing JSON files"):

    file_path = os.path.join(folder, file)

    # Skip empty files
    if os.path.getsize(file_path) == 0:
        print(f"⚠️ Skipping EMPTY file: {file}")
        continue

    # ---------- Load JSON ----------
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text.startswith("{") and not text.startswith("["):
                print(f"⚠️ Invalid JSON structure, skipping: {file}")
                continue
            try:
                data = json.loads(text)
            except:
                print(f"⚠️ CORRUPTED JSON, skipping: {file}")
                continue
    except:
        print(f"⚠️ Cannot open file: {file}")
        continue

    # convert dict → list
    if isinstance(data, dict):
        data = [data]

    # convert into dataframe
    df = pd.DataFrame(data)
    df["source_file"] = file

    # ensure required columns exist
    for col in ["question", "answer", "p_answer"]:
        if col not in df.columns:
            df[col] = None

    # drop rows with missing essential values
    df = df.dropna(subset=["question", "answer", "p_answer"])

    # detect edit type
    df["type"] = df["question"].apply(get_edit_type)

    # correctness
    df["correct"] = (
        df["p_answer"].astype(str).str.strip().str.lower()
        ==
        df["answer"].astype(str).str.strip().str.lower()
    )

    # merge for full dataset summary
    all_dfs.append(df)

    # --------- per-file accuracy summary ----------
    file_result = {
        "file_name": file,
        "num_samples": len(df),
        "overall_accuracy": round(df["correct"].mean() * 100, 2)
    }

    # accuracy by edit type
    for t in ["substitute", "remove", "add", "swap", "other"]:
        type_df = df[df["type"] == t]
        if len(type_df) > 0:
            acc = type_df["correct"].mean() * 100
        else:
            acc = 0
        file_result[f"acc_{t}"] = round(acc, 2)

    # accuracy by difficulty
    if "difficulty" in df.columns:
        df["difficulty"] = df["difficulty"].astype(str).fillna("unknown")
        diff_acc = df.groupby("difficulty")["correct"].mean() * 100
        for d, val in diff_acc.items():
            file_result[f"acc_difficulty_{d}"] = round(val, 2)

    file_summaries.append(file_result)

# -------------------------------------------------------
# MERGED FINAL DATA (all JSONs combined)
# -------------------------------------------------------
if len(all_dfs) > 0:
    merged_df = pd.concat(all_dfs, ignore_index=True)
else:
    merged_df = pd.DataFrame()

merged_out = "merged_eval_results.json"
merged_df.to_json(merged_out, orient="records", indent=2, force_ascii=False)
print(f"\n📁 Saved merged results → {merged_out}")

# -------------------------------------------------------
# SAVE PER-FILE ACCURACY TABLE
# -------------------------------------------------------
summary_df = pd.DataFrame(file_summaries)
summary_out = "accuracies_per_file.csv"
summary_df.to_csv(summary_out, index=False)
print(f"📊 Saved per-file accuracies → {summary_out}")

print("\n✅ Finished! All accuracies computed successfully.")
