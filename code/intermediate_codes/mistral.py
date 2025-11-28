import json
import re
from tqdm import tqdm
import ollama
import pandas as pd
from ollama import Client
client = Client(host="http://localhost:11434")
# ----------------------------------------------------
# 1. Few-Shot Examples (same, but stronger add/swap)
# ----------------------------------------------------
FEW_SHOT_EXAMPLES = """
Examples:
Substitute 'a' with 'e' in 'cat' → "cet"
Remove 'u' after every 't' in 'structure' → "structre"

# Optimized ADD and SWAP examples
Swap 'i' and 'g' in 'pig' → "gip"
Swap 'a' and 'n' in 'banana' → "nanaba"
Add 'o' after 'i' in 'pin' → "poin"
Add 'a' after 'n' in 'banana' → "banaanaa"

Now follow the same pattern for the next instruction.
"""

# ----------------------------------------------------
# 2. Prompt Template (unchanged)
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
# 3. Load Dataset from JSONL
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
# 5. Operation-Specific Hint (only improved for add/swap)
# ----------------------------------------------------
def get_operation_hint(question):
    q = question.lower()
    if "substitute" in q:
        return "Hint: This is a substitution task. Replace one character with another."
    elif "remove" in q:
        return "Hint: This is a removal task. Delete the specified character carefully."
    elif "swap" in q:
        return ("Hint: This is a swap task. Exchange both characters wherever they appear — "
                "each occurrence of the first becomes the second, and vice versa.")
    elif "add" in q:
        return ("Hint: This is an addition task. Insert the given character after "
                "every occurrence of the specified one in the word.")
    else:
        return "Hint: Perform the described edit carefully."

# ----------------------------------------------------
# 6. Warm up model (changed to mistral on CPU)
# ----------------------------------------------------
print("\n🔄 Loading model into memory (CPU mode)...")
_ = client.generate(
    model="mistral:latest",
    prompt="Ready?",
    options={"temperature": 0, "num_predict": 20, "num_gpu": 0}
)
print("✅ Model ready.\n")

# ----------------------------------------------------
# 7. Evaluation Loop (same except tuned temps)
# ----------------------------------------------------
correct = 0
total = len(data)
results = []

for item in tqdm(data, desc="Evaluating 10k responses"):
    question = item["question"]
    expected = item["answer"]

    # Annotate + add hint
    question_annotated = annotate_instruction(question)
    hint = get_operation_hint(question)

    prompt_text = PROMPT_TEMPLATE.format(
        examples=FEW_SHOT_EXAMPLES.strip(),
        question=f"{question_annotated}\n\n{hint}"
    )

    # slightly boosted creativity for add/swap only
    if "swap" in question.lower():
        temp = 0.25
    elif "add" in question.lower():
        temp = 0.2
    else:
        temp = 0.0

    try:
        result = client.generate(
            model="mistral:latest",
            prompt=prompt_text,
            options={"temperature": temp, "num_predict": 100, "num_gpu": 0}
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

# Add type column for breakdown
for item in results:
    q = item["question"].lower()
    if "substitute" in q:
        item["type"] = "substitute"
    elif "remove" in q:
        item["type"] = "remove"
    elif "add" in q:
        item["type"] = "add"
    elif "swap" in q:
        item["type"] = "swap"
    else:
        item["type"] = "other"

df = pd.DataFrame(results)

print("\n📊 Accuracy by Type:")
print((df.groupby("type")["p_answer"].apply(lambda x: (x == df.loc[x.index, 'answer']).mean() * 100)).sort_index())

if "difficulty" in df.columns:
    print("\n📈 Accuracy by Difficulty Level:")
    print((df.groupby("difficulty")["p_answer"].apply(lambda x: (x == df.loc[x.index, 'answer']).mean() * 100)).sort_index())

# ----------------------------------------------------
# 9. Save Results
# ----------------------------------------------------
output_file = "mistral_test10k_addswap_cpu.json"
df.to_json(output_file, orient="records", indent=2, force_ascii=False)
print(f"\n📁 Saved results to: {output_file}")
