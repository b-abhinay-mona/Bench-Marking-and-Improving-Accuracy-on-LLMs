import json
import re
from tqdm import tqdm
import ollama
import pandas as pd
import os
os.environ["OLLAMA_NO_GPU"] = "1"
# ----------------------------------------------------
# 1. Few-Shot Examples (separate by type)
# ----------------------------------------------------
FEW_SHOT_EXAMPLES = {
    "substitute": """
Examples:
Substitute 'a' with 'e' in 'cat' → "cet"
Substitute 'i' with 'a' in 'bird' → "bard"
Substitute 'o' with 'u' in 'dog' → "dug"
""",

    "remove": """
Examples:
Remove 'u' after every 't' in 'structure' → "structre"
Remove 'e' from 'cheese' → "ches"
Remove 'a' from 'banana' → "bnn"
""",

    "add": """
Examples:
Add 'o' after 'i' in 'pin' → "poin"
Add 'a' after 'n' in 'banana' → "banaanaa"
Add 'e' after 'l' in 'ball' → "ballel"
""",

    # note: "swap" examples are rephrased as two substitutes
    "swap": """
Examples (rephrased as substitutes):
Substitute 'p' with 'n' in 'pan' and substitute 'n' with 'p' in 'pan' → "nap"
Substitute 'a' with 'b' in 'bat' and substitute 'b' with 'a' in 'bat' → "abt"
Substitute 'i' with 'g' in 'pig' and substitute 'g' with 'i' in 'pig' → "gip"
"""
}

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
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid line: {line[:80]}... ({e})")

print(f"✅ Loaded {len(data)} examples from {input_file}")

# ----------------------------------------------------
# 4. Annotate Instruction (same)
# ----------------------------------------------------
def annotate_instruction(question: str) -> str:
    quoted = re.findall(r"'([^']+)'", question)
    if len(quoted) < 2:
        return question

    src_char = quoted[0]
    target_word = quoted[-1]
    q_lower = question.lower()

    if "substitute" in q_lower or "remove" in q_lower:
        parts = target_word.split(src_char)
        annotated_target = f" {src_char} ".join(parts)
    elif "swap" in q_lower or "add" in q_lower:
        annotated_target = " ".join(list(target_word))
    else:
        annotated_target = target_word

    annotated = question.replace(f"'{target_word}'", annotated_target)
    return annotated

# ----------------------------------------------------
# 5. Operation-Specific Hint
# ----------------------------------------------------
def get_operation_hint(op_type):
    if op_type == "substitute":
        return "Hint: Replace the specified character with another."
    elif op_type == "remove":
        return "Hint: Delete the specified character carefully."
    elif op_type == "swap":
        return ("Hint: This is equivalent to performing two substitutions — "
                "each occurrence of the first character becomes the second, and vice versa.")
    elif op_type == "add":
        return "Hint: Insert the given character after every occurrence of the specified one."
    else:
        return "Hint: Perform the described edit carefully."

# ----------------------------------------------------
# 6. Prepare Model
# ----------------------------------------------------
print("\n🔄 Loading model into memory (GPU if available)...")
_ = ollama.generate(
    model="llama3.2:latest",
    prompt="Ready?",
    options={"temperature": 0, "num_predict": 20}
)
print("✅ Model ready.\n")

# ----------------------------------------------------
# 7. Evaluation Loop (uses 'type' and modifies swap)
# ----------------------------------------------------
correct = 0
total = len(data)
results = []

for item in tqdm(data, desc="Evaluating responses"):
    question = item["question"]
    expected = item["answer"]
    op_type = item.get("type", "").lower()

    # For swap → transform into two substitute instructions
    if op_type == "swap":
        src = item.get("source", "")
        tgt = item.get("target", "")
        word = item.get("word", "")
        # Build a new instruction combining two substitutes
        question = f"Substitute '{src}' with '{tgt}' in '{word}' and substitute '{tgt}' with '{src}' in '{word}'."

    # Annotate + hint
    question_annotated = annotate_instruction(question)
    hint = get_operation_hint(op_type)

    # Get few-shot examples based on type
    examples = FEW_SHOT_EXAMPLES.get(op_type, FEW_SHOT_EXAMPLES["substitute"]).strip()

    prompt_text = PROMPT_TEMPLATE.format(
        examples=examples,
        question=f"{question_annotated}\n\n{hint}"
    )

    # temperature tuning
    if op_type == "swap":
        temp = 0.25
    elif op_type == "add":
        temp = 0.2
    else:
        temp = 0.0

    try:
        result = ollama.generate(
            model="llama3.2:latest",
            prompt=prompt_text,
            options={"temperature": temp, "num_predict": 100}
        )
        response = result["response"].strip()
    except Exception as e:
        print(f"\n⚠️ Error on: {question}\n{e}")
        response = ""

    match = re.search(r'"p_answer"\s*:\s*"([^"]+)"', response)
    p_answer = match.group(1).strip() if match else ""
    p_answer = p_answer.replace(" ", "")

    item["p_answer"] = p_answer
    item["raw_response"] = response
    item["annotated_question"] = question_annotated

    if p_answer == expected:
        correct += 1

    results.append(item)

# ----------------------------------------------------
# 8. Accuracy Summary
# ----------------------------------------------------
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n✅ Overall Accuracy: {correct}/{total} = {accuracy:.2f}%")

df = pd.DataFrame(results)

print("\n📊 Accuracy by Type:")
print((df.groupby("type")["p_answer"]
       .apply(lambda x: (x == df.loc[x.index, 'answer']).mean() * 100))
       .sort_index())

if "difficulty" in df.columns:
    print("\n📈 Accuracy by Difficulty Level:")
    print((df.groupby("difficulty")["p_answer"]
           .apply(lambda x: (x == df.loc[x.index, 'answer']).mean() * 100))
           .sort_index())

# ----------------------------------------------------
# 9. Save Results
# ----------------------------------------------------
output_file = "llama3_2_test10k_swap_as_substitute.json"
df.to_json(output_file, orient="records", indent=2, force_ascii=False)
print(f"\n📁 Saved results to: {output_file}")
