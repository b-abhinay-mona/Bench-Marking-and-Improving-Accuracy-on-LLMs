import json
import pandas as pd
import numpy as np
import re
import os

# ==============================
# CONFIG
# ==============================
initial_file = "llama32latest_2.json"
final_file   = "llama3_2_test10k_addswap_finetuned.json"

# Output files
os.makedirs("cfm_results", exist_ok=True)
all_csv      = "cfm_results/cfm_all_samples.csv"
edit_csv     = "cfm_results/cfm_edit_type_summary.csv"
source_csv   = "cfm_results/cfm_char_source_summary.csv"
target_csv   = "cfm_results/cfm_char_target_summary.csv"
length_csv   = "cfm_results/cfm_length_summary.csv"

# ==============================
# Levenshtein Distance Function
# ==============================
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = np.zeros((la + 1, lb + 1), dtype=int)
    dp[:, 0] = np.arange(la + 1)
    dp[0, :] = np.arange(lb + 1)

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,
                dp[i, j-1] + 1,
                dp[i-1, j-1] + cost
            )
    return dp[la, lb]

# CFM Metric
def compute_cfm(pred: str, true: str) -> float:
    pred = re.sub(r"\s+", "", str(pred))
    true = re.sub(r"\s+", "", str(true))
    max_len = max(len(pred), len(true))
    lev = levenshtein(pred, true)
    if max_len == 0:
        return 1.0
    return max(min(1.0 - (lev / max_len), 1.0), 0.0)

# ==============================
# Load JSON (array or jsonl)
# ==============================
def load_json(path):
    text = open(path, "r", encoding="utf-8").read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]

# ==============================
# Prepare DataFrame
# ==============================
def prepare_df(path, label):
    data = load_json(path)
    df = pd.DataFrame(data)

    for c in ["p_answer", "answer", "type", "source", "target", "length"]:
        if c not in df.columns:
            df[c] = None

    df["p_answer_clean"] = df["p_answer"].astype(str).apply(lambda s: re.sub(r"\s+","",s))
    df["answer_clean"] = df["answer"].astype(str).fillna("")

    df["exact"] = (df["p_answer_clean"] == df["answer_clean"])
    df["cfm"] = df.apply(lambda r: compute_cfm(r["p_answer_clean"], r["answer_clean"]), axis=1)
    df["Model"] = label

    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    return df

df_i = prepare_df(initial_file, "Initial")
df_f = prepare_df(final_file, "Final")
df = pd.concat([df_i, df_f], ignore_index=True)

# ==============================
# SAVE FULL SAMPLE REPORT
# ==============================
df.to_csv(all_csv, index=False)
print(f"\n✔ Saved full per-sample results → {all_csv}")

# ==============================
# SUMMARY TABLES
# ==============================
# Edit Type Summary
edit_summary = df.groupby(["Model", "type"]).agg(
    exact_accuracy=("exact","mean"),
    mean_cfm=("cfm","mean")
).reset_index()
edit_summary.to_csv(edit_csv, index=False)
print(f"✔ Saved edit-type summary → {edit_csv}")

# Source Character Summary
source_summary = df.groupby(["Model", "source"]).agg(
    exact_accuracy=("exact","mean"),
    mean_cfm=("cfm","mean")
).reset_index()
source_summary.to_csv(source_csv, index=False)
print(f"✔ Saved source-char summary → {source_csv}")

# Target Character Summary
target_summary = df.groupby(["Model", "target"]).agg(
    exact_accuracy=("exact","mean"),
    mean_cfm=("cfm","mean")
).reset_index()
target_summary.to_csv(target_csv, index=False)
print(f"✔ Saved target-char summary → {target_csv}")

# Word Length Summary
length_summary = df.groupby(["Model", "length"]).agg(
    exact_accuracy=("exact","mean"),
    mean_cfm=("cfm","mean")
).reset_index()
length_summary.to_csv(length_csv, index=False)
print(f"✔ Saved word-length summary → {length_csv}")

print("\n🎯 All CFM result tables saved successfully!")
