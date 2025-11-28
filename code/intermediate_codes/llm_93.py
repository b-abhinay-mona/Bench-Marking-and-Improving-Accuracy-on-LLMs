import json
import re
from tqdm import tqdm
import ollama
import pandas as pd
from ollama import Client
client = Client(host="http://localhost:11434")
# ----------------------------------------------------
# 1. Type-Specific Few-Shot Examples
# ----------------------------------------------------
FEW_SHOT_EXAMPLES = {
    "substitute": """
Examples:
Substitute 'a' with 'e' in 'cat' → "cet"
Substitute 'r' with 'l' in 'mirror' → "millol"
Substitute 'o' with 'a' in 'door' → "daar"
Now follow the same substitution logic for the next instruction.
""",
    "remove": """
Examples:
Remove 'u' after every 't' in 'structure' → "structre"
Remove 'r' from 'parrot' → "paot"
Remove 'e' from 'cheese' → "ches"
Now follow the same removal logic for the next instruction.
""",
    "add": """
Examples:
Add 'o' after 'r' in 'parrot' → "paroroot"
Add 'a' after 'n' in 'banana' → "banaanaa"
Add 'e' after 'r' in 'car' → "caer"
Remember: Add the given character after *every* occurrence of the specified letter.
""",
    "swap": """
Examples:
To swap two letters, perform two substitutions:
Swap 'a' and 'n' in 'banana' → substitute 'a' with 'n' and 'n' with 'a' → "nanaba"
Swap 'o' and 'r' in 'parrot' → substitute 'o' with 'r' and 'r' with 'o' → "paoort"
Swap 'i' and 'g' in 'pig' → substitute 'i' with 'g' and 'g' with 'i' → "gip"
Now follow this two-step substitution pattern for the next swap.
""",
}

# ----------------------------------------------------
# 2. Prompt Template (unchanged)
# ----------------------------------------------------
PROMPT_TEMPLATE = """
You are a precise text editing assistant.

{examples}

Instruction: {question}

Think carefully about each letter position before editing.
Apply the edit to all relevant occurrences.
Then output the final transformed word only.

Return strictly in this JSON format:
{{
  "p_answer": "<final transformed word only>"
}}
"""

# ----------------------------------------------------
# 3. Load Dataset
# ----------------------------------------------------
input_file = "dataset_2.5k_4types.jsonl"

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
# 4. Edit-Aware Split Function (same)
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
# 5. Type Detection
# ----------------------------------------------------
def detect_type(question: str):
    q = question.lower()
    if "substitute" in q:
        return "substitute"
    elif "remove" in q:
        return "remove"
    elif "add" in q:
        return "add"
    elif "swap" in q:
        return "swap"
    else:
        return "other"

# ----------------------------------------------------
# 6. Operation-Specific Hint
# ----------------------------------------------------
def get_operation_hint(edit_type: str):
    if edit_type == "substitute":
        return "Hint: Replace all source characters with the target ones."
    elif edit_type == "remove":
        return "Hint: Remove all occurrences as described."
    elif edit_type == "add":
        return "Hint: Add the new character after every instance of the given letter."
    elif edit_type == "swap":
        return "Hint: Perform two substitutions — swap the two letters in all their positions."
    else:
        return "Hint: Perform the edit as instructed."

# ----------------------------------------------------
# 7. Warm up Model
# ----------------------------------------------------
print("\n🔄 Loading llama3.2:latest into memory (GPU if available)...")
_ = client.generate(
    model="llama3.2:latest",
    prompt="Ready?",
    options={"temperature": 0, "num_predict": 20}
)
print("✅ Model ready.\n")

# ----------------------------------------------------
# 8. Evaluation Loop (type-aware few-shot prompts)
# ----------------------------------------------------
correct = 0
total = len(data)
results = []

for item in tqdm(data, desc="Evaluating type-aware responses"):
    question = item["question"]
    expected = item["answer"]
    edit_type = detect_type(question)

    # Get type-specific examples and hint
    examples = FEW_SHOT_EXAMPLES.get(edit_type, FEW_SHOT_EXAMPLES["substitute"])
    hint = get_operation_hint(edit_type)
    annotated_q = annotate_instruction(question)

    # Build final prompt
    prompt_text = PROMPT_TEMPLATE.format(
        examples=examples.strip(),
        question=f"{annotated_q}\n\n{hint}"
    )

    # tuned temperatures
    if edit_type == "swap":
        temp = 0.3
    elif edit_type == "add":
        temp = 0.25
    else:
        temp = 0.0

    try:
        result = client.generate(
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
    item["annotated_question"] = annotated_q
    item["type"] = edit_type
    item["is_correct"] = (p_answer == expected)

    if item["is_correct"]:
        correct += 1

    results.append(item)

# ----------------------------------------------------
# 9. Accuracy Summary
# ----------------------------------------------------
df = pd.DataFrame(results)
overall = df["is_correct"].mean() * 100
print(f"\n✅ Overall Accuracy: {overall:.2f}%")

print("\n📊 Accuracy by Type:")
print((df.groupby("type")["is_correct"].mean() * 100).sort_index())

if "difficulty" in df.columns:
    print("\n📈 Accuracy by Difficulty Level:")
    print((df.groupby("difficulty")["is_correct"].mean() * 100).sort_index())

# ----------------------------------------------------
# 10. Save Results
# ----------------------------------------------------
output_file = "llama3_2_test10k_typeaware_optimized.json"
df.to_json(output_file, orient="records", indent=2, force_ascii=False)
print(f"\n📁 Saved results to: {output_file}")
