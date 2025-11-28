import json
import re
from tqdm import tqdm
from ollama import Client
# Create a single Ollama client (new API, stable on Windows)
client = Client(host="http://localhost:11434")
# -----------------------------
# 1. Simple prompt template
# -----------------------------
PROMPT_TEMPLATE = """
You are a precise text editing assistant.

Instruction: {question}

Return your final result strictly in this JSON format:
{{
  "p_answer": "<final transformed word only>"
}}
"""

# -----------------------------
# 2. Load the dataset
# -----------------------------
input_file = "test_10k.jsonl"

data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# -----------------------------
# 3. Warm up the model (GPU ready)
# -----------------------------
print("🔄 Loading model into memory (GPU if available)...")
_ = client.generate(
    model="qwen2.5:3b",
    prompt="Ready?",
    options={"temperature": 0, "num_predict": 20}
)
print("✅ Model ready.\n")

# -----------------------------
# 4. Evaluation loop (GPU)
# -----------------------------
correct = 0
total = len(data)
results = []

for item in tqdm(data, desc="Evaluating responses"):
    question = item["question"]
    expected = item["answer"]

    prompt_text = PROMPT_TEMPLATE.format(question=question)

    try:
        # Ollama will use GPU automatically if available
        result = client.generate(
            model="qwen2.5:3b",
            prompt=prompt_text,
            options={"temperature": 0, "num_predict": 100}  # keep same
        )
        response = result["response"].strip()
    except Exception as e:
        print(f"\nError on: {question}\n{e}")
        response = ""

    # Extract p_answer from JSON
    match = re.search(r'"p_answer"\s*:\s*"([^"]+)"', response)
    if match:
        p_answer = match.group(1).strip()
    else:
        p_answer = ""

    # Save prediction
    item["p_answer"] = p_answer
    item["raw_response"] = response

    # Evaluate
    if p_answer == expected:
        correct += 1

    results.append(item)

# -----------------------------
# 5. Accuracy
# -----------------------------
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n✅ Accuracy: {correct}/{total} = {accuracy:.2f}%")

# -----------------------------
# 6. Save results
# -----------------------------
output_file = "qwen_intial.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"📁 Saved results to: {output_file}")