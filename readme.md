# FMGAI Project: Character-Level Text Manipulation with LLMs

## Project Overview
This project benchmarks and improves Large Language Model (LLM) performance on deterministic character-level text manipulation tasks, including:

- Add a character
- Remove a character
- Substitute one character with another
- Swap two characters

We introduce a character-level preprocessing strategy (explicit spacing) combined with structured few-shot prompting and a multi-step hybrid correction for swap tasks. These methods significantly improve exact-match performance without modifying model weights.

Final accuracy improvement: 4.51% → 30.70%  
Dataset: 10,000 character-edit instructions

---

## Repository Structure

```plaintext
genai_expmt/
│
├── FMGAI_Project_Report.pdf               # Final IEEE-style project report
├── FM_GAI_PROJECT.zip                     # Full submission archive
├── requirements.txt                       # Python dependencies
├── .env                                   # Local environment config
│
├── code/
│   └── final_codes/
│       ├── initial.py                     # Baseline zero-shot evaluation
│       ├── final.py                       # Final prompt strategy evaluation
│       ├── swap_multi_step_llm.py         # Multi-step hybrid swap execution
│       ├── p_tuning_v2.ipynb              # Soft prompt tuning (P-Tuning v2)
│       ├── ptuning_eval.ipynb             # Soft prompt evaluation
│       └── xprompt.ipynb                  # Soft prompt tuning (XPrompt)
│
├── dataset/
│   ├── test_100.jsonl                     # Debug mini evaluation set
│   └── test_10k.jsonl                     # Full benchmark dataset
│
├── results(json)/
│   ├── accuracies_per_file.csv            # Evaluation summary
│   ├── initial_final_result/              # JSON predictions (initial vs final)
│   └── intermediate_results/              # Partial evaluation traces
│
├── Levenshtein_metric_results/
│   ├── cfm_edit_type_summary.csv
│   ├── cfm_char_source_summary.csv
│   ├── cfm_char_target_summary.csv
│   ├── cfm_length_summary.csv
│   └── cfm_results_combined.csv
│
├── prompts/
│   ├── base_prompt.txt                    # Baseline prompt
│   ├── fewshot_prompts/                   # Few-shot task examples
│   └── opspecific_prompts/                # Operation-specific prompts
│
├── plots/                                 # Final visual analysis
│   ├── accuracy_across_edit_types.png
│   ├── accuracy_vs_word_length.png
│   ├── source_accuracy_character.png
│   ├── target_accuracy_character.png
│   ├── CFM_wordlength.png
│   ├── cmf_exactness.png
│   ├── CMF_source_character_accuracy.png
│   └── … additional internal visuals
│
└── venv/                                   # Optional local Python environment
````

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Baseline Evaluation

```bash
python3 code/final_codes/initial.py
```

### 3. Final Improved Prompt Strategy

```bash
python3 code/final_codes/final.py
```

### 4. Multi-Step Hybrid Swap Execution

```bash
python3 code/final_codes/swap_multi_step_llm.py
```

### 5. Levenshtein Similarity Evaluation

```bash
python3 Levenshtein_metric_results/Levenshtein.py
```

Note: Requires local Ollama installation with `llama3.2:latest`

---

## Evaluation Metrics

| Metric                       | Purpose                        | Importance                            |
| ---------------------------- | ------------------------------ | ------------------------------------- |
| Exact-Match Accuracy         | Measures fully correct outputs | Required for deterministic edits      |
| Levenshtein Similarity Score | Measures closeness to correct  | Shows partial correctness improvement |

---

## Key Contributions

* First systematic evaluation of LLMs on deterministic character edits
* Tokenization-aware preprocessing with strong accuracy gains
* Hybrid execution strategy eliminates generative errors for swap tasks
* Levenshtein-based similarity demonstrates deeper model understanding

---

## Future Work

* Extend from isolated words to sentence-level editing
* Handle substring and contextual morphological changes
* Improve swap/add using extended multi-step logic
* Low-compute adapter tuning for open-source models

---

## Authors

| Name                   | Institute   | Email                                             |
| ---------------------- | ----------- | ------------------------------------------------- |
| Bisamalla Abhinay      | IIT Jodhpur | [b22ai012@iitj.ac.in](mailto:b22ai012@iitj.ac.in) |
| Cheruvu Mohammad Fazil | IIT Jodhpur | [b22ai046@iitj.ac.in](mailto:b22ai046@iitj.ac.in) |

---

## Acknowledgment

We thank **Dr. Mayank Vatsa Sir** for guidance, feedback, and discussion of the research motivation.

---

## License

Academic use only — IIT Jodhpur coursework project.
