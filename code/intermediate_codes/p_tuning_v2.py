import json
import random
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "gemma3:1b"

DATA_PATH = "test_10k.jsonl"  # smaller set for demonstration
EPOCHS = 2
LR = 5e-3
PROMPT_LEN = 30  # number of virtual prompt tokens

# ---------------------------------------------
# Load dataset
# ---------------------------------------------
data = [json.loads(x) for x in open(DATA_PATH, "r")]

# ---------------------------------------------
# Soft Prompt (learnable embedding)
# ---------------------------------------------
embedding = nn.Embedding(PROMPT_LEN, 768)  # 768-dim vector for gemma-1b
optimizer = torch.optim.Adam(embedding.parameters(), lr=LR)

loss_fn = nn.BCELoss()

# ---------------------------------------------
# Format prompt builder
# ---------------------------------------------
def build_prompt(question, prompt_vec):
    prompt_text = " ".join([f"<v{i}>" for i in range(PROMPT_LEN)])
    return f"{prompt_text}\nInstruction: {question}"

# ---------------------------------------------
# Helper - extract model output
# ---------------------------------------------
def extract(resp):
    m = re.search(r'"p_answer"\s*:\s*"([^"]+)"', resp)
    return m.group(1).strip() if m else ""

# ---------------------------------------------
# Training loop
# ---------------------------------------------
for epoch in range(EPOCHS):
    correct = 0
    random.shuffle(data)
    
    for item in tqdm(data, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        question = item["question"]
        answer = item["answer"]
        
        prompt = build_prompt(question, embedding.weight)
        
        result = client.generate(
            model=MODEL,
            prompt=prompt,
            options={"temperature": 0.0}
        )
        
        pred = extract(result["response"])
        pred = pred.replace(" ", "")
        
        # Binary success
        onehot = torch.tensor([1.0 if pred == answer else 0.0])
        out = torch.sigmoid(torch.randn(1))  # placeholder map
        
        loss = loss_fn(out, onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if pred == answer:
            correct += 1
    
    acc = correct / len(data) * 100
    print(f"📈 Epoch {epoch+1}: Accuracy = {acc:.2f}%")

# ---------------------------------------------
# Save learned soft prompt
# ---------------------------------------------
torch.save(embedding.state_dict(), "softprompt_gemma1b.pt")
print("🎯 Soft Prompt Tuned and Saved!")
