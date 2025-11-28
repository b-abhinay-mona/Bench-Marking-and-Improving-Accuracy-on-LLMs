import json
import re
from tqdm import tqdm
from ollama import Client
import pandas as pd

# Create a single Ollama client (new API, stable on Windows)
client = Client(host="http://localhost:11434")

# ----------------------------------------------------
# 1. Few-Shot Examples (only for swap → rephrased as substitutes)
# ----------------------------------------------------
FEW_SHOT_EXAMPLES_SWAP = """
Examples:
Substitute 'p' with 'n' in 'pan' and substitute 'n' with 'p' in 'pan' → "nap"
Substitute 'i' with 'g' in 'pig' and substitute 'g' with 'i' in 'pig' → "gip"
"""

# ----------------------------------------------------
# 2. Prompt Template
# ----------------------------------------------------
PROMPT_TEMPLATE = """
You are a precise text editing assistant.

{examples}

Instruction: {question}

Think carefully about each letter position before editing.
Then output the final transformed word only.

Return strictly in this JSON format:
{{
  "p_answer": "<final transformed word only>"
}}
"""

# ----------------------------------------------------
# 3. Load Dataset
# ----------------------------------------------------
input_file = "test_10k.jsonl"

data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                entry = json.loads(line)
                # Only keep swap type examples
                if entry.get("type", "").lower() == "swap":
                    data.append(entry)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid line: {line[:80]}... ({e})")

print(f"✅ Loaded {len(data)} swap examples from {input_file}")

# ----------------------------------------------------
# 4. Hint and Prompt Construction
# ----------------------------------------------------
def build_swap_prompt(entry):
    src = entry.get("source", "")
    tgt = entry.get("target", "")
    word = entry.get("word", "")
    # Rephrase swap as two substitutions
    question = (
        f"Substitute '{src}' with '{tgt}' in '{word}' "
        f"and substitute '{tgt}' with '{src}' in '{word}'."
    )
    hint = (
        "Hint: Perform both substitutions carefully. "
        "Every occurrence of the first character becomes the second, and vice versa."
    )
    examples = FEW_SHOT_EXAMPLES_SWAP.strip()
    prompt_text = PROMPT_TEMPLATE.format(
        examples=examples,
        question=f"{question}\n\n{hint}"
    )
    return prompt_text

# ----------------------------------------------------
# 5. Run Inference (swap only)
# ----------------------------------------------------
print("\n🔄 Loading model into memory (CPU mode if GPU unavailable)...")
_ = client.generate(
    model="llama3.2:latest",  # smaller, lighter model
    prompt="Ready?",
    options={"temperature": 0, "num_predict": 20}
)
print("✅ Model ready.\n")

correct = 0
total = len(data)
results = []

for item in tqdm(data, desc="Evaluating swap responses"):
    expected = item["answer"]
    prompt_text = build_swap_prompt(item)

    try:
        result = client.generate(
            model="llama3.2:3b",
            prompt=prompt_text,
            options={"temperature": 0.2, "num_predict": 100}
        )
        response = result["response"].strip()
    except Exception as e:
        print(f"\n⚠️ Error on: {item['question']}\n{e}")
        response = ""

    match = re.search(r'"p_answer"\s*:\s*"([^"]+)"', response)
    p_answer = match.group(1).strip() if match else ""
    p_answer = p_answer.replace(" ", "")

    item["p_answer"] = p_answer
    item["raw_response"] = response

    if p_answer == expected:
        correct += 1

    results.append(item)

# ----------------------------------------------------
# 6. Accuracy Summary
# ----------------------------------------------------
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n✅ Swap Accuracy: {correct}/{total} = {accuracy:.2f}%")

df = pd.DataFrame(results)
output_file = "llama3_1_swap_eval.json"
df.to_json(output_file, orient="records", indent=2, force_ascii=False)
print(f"\n📁 Saved swap-only results to: {output_file}")
