# import json
# import re
# from tqdm import tqdm
# from langchain_ollama import OllamaLLM
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from langchain.prompts import PromptTemplate


# # -----------------------------
# # 1. Define Pydantic model for structured output
# # -----------------------------
# class LLMOutput(BaseModel):
#     p_answer: str = Field(description="The predicted answer from the LLM")


# # -----------------------------
# # 2. Define the Output Parser
# # -----------------------------
# parser = PydanticOutputParser(pydantic_object=LLMOutput)


# # -----------------------------
# # 3. Initialize Ollama model
# # -----------------------------
# llm = OllamaLLM(
#     model="llama3.2:latest",
#     temperature=1,
#     num_predict=100
# )


# # -----------------------------
# # 4. Define an improved concise prompt template
# # -----------------------------
# substitute_prompt = PromptTemplate(
#     template=(
# """
# You are a precise text editor specialized in **Substitute** operations.

# ### Task Description
# When instructed to *Substitute 'X' with 'Y' in 'WORD'*, replace every occurrence of X with Y **exactly as written**.
# - Replace all non-overlapping matches.
# - Be case-sensitive.
# - Base changes on the original word only.

# ### How to Do It
# 1. Identify every occurrence of X in WORD.
# 2. Replace each X with Y.
# 3. Do not modify any other character.
# 4. Output only the final modified word.

# ### Common Mistakes & Fixes
# -  Replacing only the first occurrence →  Replace **all** occurrences.
# -  Ignoring case →  Match case exactly.
# -  Adding spaces, punctuation, or quotes →  Output the bare word only.

# ### Examples
# Instruction: Substitute 'r' with 'l' in 'personal'  
# Output: pelsonal  

# Instruction: Substitute 'a' with 'o' in 'banana'  
# Output: bonono  

# Instruction: Substitute 'x' with 'y' in 'xylophone'  
# Output: yylophone  

# ### Now follow the same pattern:

# Instruction: {question}

# Return your answer strictly in JSON:
# {{
#   "p_answer": "<final transformed word>"
# }}
# {format_instructions}
# """
#     ),
#     input_variables=["question"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# swap_prompt = PromptTemplate(
#     template=(
# """
# You are a precise text editor specialized in **Swap** operations.

# ### Task Description
# When instructed to *Swap 'A' and 'B' in 'WORD'*, exchange all occurrences of A and B **simultaneously**.
# That means each 'A' becomes 'B' and each 'B' becomes 'A', all at once.

# ### How to Do It
# 1. Work from the original string — do not let earlier swaps affect later ones.
# 2. For each character:
#    - If it's A, output B.
#    - If it's B, output A.
#    - Otherwise, keep it unchanged.
# 3. Output only the transformed word.

# ### Common Mistakes & Fixes
# -  Doing two sequential replacements (A→B then B→A) →  Do it simultaneously.
# -  Forgetting to swap all occurrences →  Apply to all.
# -  Changing other characters →  Only A and B are swapped.
# -  Adding explanation text →  Output only the final word.

# ### Examples
# Instruction: Swap 'i' and 'g' in 'imaginary'  
# Output: gmaignary  

# Instruction: Swap 'a' and 'b' in 'abracadabra'  
# Output: baracadbabra  

# Instruction: Swap 'x' and 'y' in 'syntax'  
# Output: syxtan  

# ### Now follow the same pattern:

# Instruction: {question}

# Return your answer strictly in JSON:
# {{
#   "p_answer": "<final transformed word>"
# }}
# {format_instructions}
# """
#     ),
#     input_variables=["question"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )


# add_prompt = PromptTemplate(
#     template=(
# """
# You are a precise text editor specialized in **Add** operations.

# ### Task Description
# When instructed to *Add 'S' after (or before) 'T' in 'WORD'*, insert the substring S at that exact position for every matching occurrence of T (unless specified otherwise).

# ### How to Do It
# 1. Read carefully whether to insert *after* or *before*.
# 2. Scan the word left to right.
# 3. Whenever you find T:
#    - If "after", output T then S.
#    - If "before", output S then T.
# 4. Leave other characters unchanged.
# 5. Output only the final transformed word.

# ### Common Mistakes & Fixes
# -  Forgetting to insert at every match →  Do it for each occurrence unless told otherwise.
# -  Reversing "before" and "after" →  Follow instruction literally.
# -  Adding extra spaces or quotes →  Output only the word.

# ### Examples
# Instruction: Add 'o' after 'i' in 'curiosity'  
# Output: curioosioty  

# Instruction: Add 'x' before 'b' in 'banana'  
# Output: xbanana  

# Instruction: Add 's' after 'e' in 'tree'  
# Output: trees  

# ### Now follow the same pattern:

# Instruction: {question}

# Return your answer strictly in JSON:
# {{
#   "p_answer": "<final transformed word>"
# }}
# {format_instructions}
# """
#     ),
#     input_variables=["question"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )


# remove_prompt = PromptTemplate(
#     template=(
# """
# You are a precise text editor specialized in **Remove** operations.

# ### Task Description
# When instructed to *Remove 'X' after/before/in 'WORD'*, delete that substring exactly as specified.
# - Be literal and case-sensitive.
# - Apply to all occurrences unless otherwise stated.

# ### How to Do It
# 1. Identify where X appears based on the rule (e.g., "after every 't'").
# 2. Remove only those X’s.
# 3. Leave everything else unchanged.
# 4. Output only the final transformed word.

# ### Common Mistakes & Fixes
# -  Removing all X’s globally →  Remove only when the condition (“after t”) is met.
# -  Ignoring the condition order →  Follow “after”, “before”, etc. literally.
# -  Leaving extra spaces →  Output only the clean, continuous word.

# ### Examples
# Instruction: Remove 'u' after every 't' in 'structure'  
# Output: structre  

# Instruction: Remove 'a' after every 'b' in 'bababa'  
# Output: bbba  

# Instruction: Remove 'e' in 'cheese'  
# Output: chs  

# ### Now follow the same pattern:

# Instruction: {question}

# Return your answer strictly in JSON:
# {{
#   "p_answer": "<final transformed word>"
# }}
# {format_instructions}
# """
#     ),
#     input_variables=["question"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# # -----------------------------
# # 5. Load the JSONL dataset
# # -----------------------------
# input_file = "test_10k.jsonl"
# data = []
# with open(input_file, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             data.append(json.loads(line))


# # -----------------------------
# # 6. Evaluate with progress bar
# # -----------------------------
# correct = 0
# total = len(data)

# prompt_map = {
#     "substitute": substitute_prompt,
#     "swap": swap_prompt,
#     "add": add_prompt,
#     "remove": remove_prompt,
# }

# for item in tqdm(data, desc="Evaluating LLM responses"):
#     qtype = item["type"].lower().strip()
#     question = item["question"]
#     expected = item["answer"]

#     # Pick the right template
#     current_prompt = prompt_map.get(qtype)
#     if not current_prompt:
#         print(f"Unknown type: {qtype}")
#         continue

#     _prompt = current_prompt.format(question=question)


#     try:
#         # Run LLM
#         response = llm.invoke(_prompt)

#         # Parse structured output
#         parsed = parser.parse(response)
#         p_answer = parsed.p_answer.strip()
#     except Exception:
#         # Regex fallback if JSON parsing fails
#         match = re.search(r'"p_answer"\s*:\s*"([^"]+)"', str(response))
#         if match:
#             p_answer = match.group(1).strip()
#         else:
#             p_answer = ""

#     # Save prediction
#     item["p_answer"] = p_answer

#     # Evaluate
#     if p_answer == expected:
#         print("\n",p_answer)
#         correct += 1

# # -----------------------------
# # 7. Calculate accuracy
# # -----------------------------
# accuracy = (correct / total) * 100 if total > 0 else 0
# print(f"\n Accuracy: {correct}/{total} = {accuracy:.2f}%")

# # -----------------------------
# # 8. Save updated JSON with predictions
# # -----------------------------
# output_file = "llama3.2_results_with_predictions.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)

# print(f"Saved results to: {output_file}")






import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# GPU-compatible Mistral model
from langchain_ollama.chat_models import ChatOllama

# Output parser and prompt template
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate



# -----------------------------
# 1. Define structured output model
# -----------------------------
class LLMOutput(BaseModel):
    p_answer: str = Field(description="The predicted answer from the LLM")

parser = PydanticOutputParser(pydantic_object=LLMOutput)


# -----------------------------
# 2. Initialize the model
# -----------------------------
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=1,
    num_predict=100
)


# -----------------------------
# 3. Define all prompts (UNCHANGED)
# -----------------------------
substitute_prompt = PromptTemplate(
    template=("""
You are a precise text editor specialized in **Substitute** operations.

### Task Description
When instructed to *Substitute 'X' with 'Y' in 'WORD'*, replace every occurrence of X with Y **exactly as written**.
- Replace all non-overlapping matches.
- Be case-sensitive.
- Base changes on the original word only.

### How to Do It
1. Identify every occurrence of X in WORD.
2. Replace each X with Y.
3. Do not modify any other character.
4. Output only the final modified word.

### Common Mistakes & Fixes
-  Replacing only the first occurrence →  Replace **all** occurrences.
-  Ignoring case →  Match case exactly.
-  Adding spaces, punctuation, or quotes →  Output the bare word only.

### Examples
Instruction: Substitute 'r' with 'l' in 'personal'  
Output: pelsonal  

Instruction: Substitute 'a' with 'o' in 'banana'  
Output: bonono  

Instruction: Substitute 'x' with 'y' in 'xylophone'  
Output: yylophone  

### Now follow the same pattern:

Instruction: {question}

Return your answer strictly in JSON:
{{
  "p_answer": "<final transformed word>"
}}
{format_instructions}
"""),
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

swap_prompt = PromptTemplate(
    template=("""
You are a precise text editor specialized in **Swap** operations.

### Task Description
When instructed to *Swap 'A' and 'B' in 'WORD'*, exchange all occurrences of A and B **simultaneously**.
That means each 'A' becomes 'B' and each 'B' becomes 'A', all at once.

### How to Do It
1. Work from the original string — do not let earlier swaps affect later ones.
2. For each character:
   - If it's A, output B.
   - If it's B, output A.
   - Otherwise, keep it unchanged.
3. Output only the transformed word.

### Common Mistakes & Fixes
-  Doing two sequential replacements (A→B then B→A) →  Do it simultaneously.
-  Forgetting to swap all occurrences →  Apply to all.
-  Changing other characters →  Only A and B are swapped.
-  Adding explanation text →  Output only the final word.

### Examples
Instruction: Swap 'i' and 'g' in 'imaginary'  
Output: gmaignary  

Instruction: Swap 'a' and 'b' in 'abracadabra'  
Output: baracadbabra  

Instruction: Swap 'x' and 'y' in 'syntax'  
Output: syxtan  

### Now follow the same pattern:

Instruction: {question}

Return your answer strictly in JSON:
{{
  "p_answer": "<final transformed word>"
}}
{format_instructions}
"""),
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

add_prompt = PromptTemplate(
    template=("""
You are a precise text editor specialized in **Add** operations.

### Task Description
When instructed to *Add 'S' after (or before) 'T' in 'WORD'*, insert the substring S at that exact position for every matching occurrence of T (unless specified otherwise).

### How to Do It
1. Read carefully whether to insert *after* or *before*.
2. Scan the word left to right.
3. Whenever you find T:
   - If "after", output T then S.
   - If "before", output S then T.
4. Leave other characters unchanged.
5. Output only the final transformed word.

### Common Mistakes & Fixes
-  Forgetting to insert at every match →  Do it for each occurrence unless told otherwise.
-  Reversing "before" and "after" →  Follow instruction literally.
-  Adding extra spaces or quotes →  Output only the word.

### Examples
Instruction: Add 'o' after 'i' in 'curiosity'  
Output: curioosioty  

Instruction: Add 'x' before 'b' in 'banana'  
Output: xbanana  

Instruction: Add 's' after 'e' in 'tree'  
Output: trees  

### Now follow the same pattern:

Instruction: {question}

Return your answer strictly in JSON:
{{
  "p_answer": "<final transformed word>"
}}
{format_instructions}
"""),
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

remove_prompt = PromptTemplate(
    template=("""
You are a precise text editor specialized in **Remove** operations.

### Task Description
When instructed to *Remove 'X' after/before/in 'WORD'*, delete that substring exactly as specified.
- Be literal and case-sensitive.
- Apply to all occurrences unless otherwise stated.

### How to Do It
1. Identify where X appears based on the rule (e.g., "after every 't'").
2. Remove only those X’s.
3. Leave everything else unchanged.
4. Output only the final transformed word.

### Common Mistakes & Fixes
-  Removing all X’s globally →  Remove only when the condition (“after t”) is met.
-  Ignoring the condition order →  Follow “after”, “before”, etc. literally.
-  Leaving extra spaces →  Output only the clean, continuous word.

### Examples
Instruction: Remove 'u' after every 't' in 'structure'  
Output: structre  

Instruction: Remove 'a' after every 'b' in 'bababa'  
Output: bbba  

Instruction: Remove 'e' in 'cheese'  
Output: chs  

### Now follow the same pattern:

Instruction: {question}

Return your answer strictly in JSON:
{{
  "p_answer": "<final transformed word>"
}}
{format_instructions}
"""),
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# -----------------------------
# 4. Load dataset
# -----------------------------
input_file = "testcases.jsonl"
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]


# -----------------------------
# 5. Prompt map
# -----------------------------
prompt_map = {
    "substitute": substitute_prompt,
    "swap": swap_prompt,
    "add": add_prompt,
    "remove": remove_prompt,
}


# -----------------------------
# 6. Worker for concurrency
# -----------------------------
def run_llm(item):
    qtype = item["type"].lower().strip()
    question = item["question"]
    expected = item["answer"]

    prompt_template = prompt_map.get(qtype)
    if not prompt_template:
        item["p_answer"] = ""
        item["raw_response"] = ""
        return item

    _prompt = prompt_template.format(question=question)

    try:
        response = llm.invoke(_prompt)
        parsed = parser.parse(response)
        p_answer = parsed.p_answer.strip()
    except Exception:
        match = re.search(r'"p_answer"\s*:\s*"([^"]+)"', str(response))
        p_answer = match.group(1).strip() if match else ""
    item["p_answer"] = p_answer
    return item


# -----------------------------
# 7. Run concurrently
# -----------------------------
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_llm, item) for item in data]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
        results.append(f.result())


# -----------------------------
# 8. Calculate accuracy
# -----------------------------
correct = sum(1 for item in results if item["p_answer"] == item["answer"])
total = len(results)
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n✅ Accuracy: {correct}/{total} = {accuracy:.2f}%")


# -----------------------------
# 9. Save output
# -----------------------------
output_file = "llama32latest_2_556.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"📁 Saved results to: {output_file}")
