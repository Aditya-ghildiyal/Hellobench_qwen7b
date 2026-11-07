# HelloBench × Qwen2-7B-Instruct, judged by Qwen2.5-14B-Instruct

Long-text generation evaluation of **Qwen2-7B-Instruct** on the [HelloBench](https://github.com/Quehry/HelloBench) benchmark (647 English prompts, 5 tasks), run end-to-end on a **free Kaggle GPU T4 × 2** with an **open-source judge** instead of GPT-4o.

```
HelloBench data ──► Qwen2-7B-Instruct writes the answers ──► Qwen2.5-14B-Instruct (4-bit) judges each checklist item ──► HelloBench score
   (647 items)        fp16, temperature 0.8, ≤ 2,048 tokens        strict prompt, 0 / 0.25 / 0.5 / 0.75 / 1          official weights, (x − 75) × 4
```

## Results

**Score** = HelloBench's rescaled score, (weighted checklist score − 75) × 4: 100 means every checklist item was fully met, 0 means every item was "mostly met" (0.75), negative means worse than that. **Raw** = weighted checklist score on a 0–100 scale before rescaling. The last two columns are the **official HelloBench numbers for the same model judged by GPT-4o**, computed from the official judgement files with this repository's scoring code.

| Task | Items | Score | Raw | Avg. words | Official score (GPT-4o judge) | Official raw |
|---|---|---|---|---|---|---|
| Open-ended QA | 200 | **10.39** | 77.60 | 739 | 46.05 | 86.51 |
| Summarization | 100 | **-103.45** | 49.14 | 424 | 7.37 | 76.84 |
| Chat | 147 | **-10.08** | 72.48 | 1062 | -6.48 | 73.38 |
| Text completion | 77 | **-40.87** | 64.78 | 996 | 5.12 | 76.28 |
| Heuristic text generation | 123 | **-7.69** | 73.08 | 678 | 16.33 | 79.08 |
| **Overall (mean of 5 tasks)** | 647 | **-30.34** | 67.42 | 780 | 13.68 | 78.42 |

All 647 items were judged; no judge output had to be dropped.

### Why the numbers are lower than the official ones

The rescaling multiplies every raw point below 75 by four, so the 11-point raw gap (67.4 vs 78.4) becomes a 44-point gap. The answers themselves are similar to the official ones in length (780 vs about 900 words on average, summaries 424 vs 452 words); the difference comes mostly from the judge. Our strict prompt tells the judge that a perfect score should be rare, and it gives half as many 1.0 scores as GPT-4o did:

| Checklist score | This run (Qwen2.5-14B judge) | Official (GPT-4o judge) |
|---|---|---|
| 1 | 932 (24%) | 1827 (47%) |
| 0.75 | 1944 (50%) | 1424 (36%) |
| 0.5 | 739 (19%) | 508 (13%) |
| 0.25 | 203 (5%) | 118 (3%) |
| 0 | 84 (2%) | 25 (1%) |

Both runs use the same 3,902 checklist questions, each applied to that run's own responses.

**What the comparison supports:** the task ranking largely agrees. Open-ended QA is the strongest task and heuristic text generation second under both judges. The biggest disagreement is **summarization**: 3rd under GPT-4o, clearly last here. The 14B judge marked many summaries down for being shorter than the "long summary" the instructions ask for and for factual slips against the source text. Long source articles (up to about 9,000 tokens) are also where a 4-bit 14B judge is least reliable.

**What it does not support:** quoting "Qwen2-7B scores −30 on HelloBench". The absolute number depends on the judge and the prompt and is not comparable to the paper's GPT-4o-judged leaderboard. Use these results to compare tasks and sub-categories within this run, or to compare other models judged with exactly this setup.

### Scores by sub-category (sorted within each task)

| Task | Sub-category | Items | Score | Raw |
|---|---|---|---|---|
| Open-ended QA | book | 19 | 17.73 | 79.43 |
| Open-ended QA | food | 20 | 16.72 | 79.18 |
| Open-ended QA | movie | 20 | 15.71 | 78.93 |
| Open-ended QA | Tech | 20 | 14.12 | 78.53 |
| Open-ended QA | Science | 21 | 13.71 | 78.43 |
| Open-ended QA | health | 20 | 10.50 | 77.63 |
| Open-ended QA | write | 20 | 8.66 | 77.17 |
| Open-ended QA | travel | 20 | 5.55 | 76.39 |
| Open-ended QA | music | 20 | 3.87 | 75.97 |
| Open-ended QA | Sport | 20 | -2.46 | 74.39 |
| Summarization | academic_article | 20 | -74.16 | 56.46 |
| Summarization | long_dialogue | 20 | -86.59 | 53.35 |
| Summarization | report | 20 | -102.23 | 49.44 |
| Summarization | blogs | 20 | -112.08 | 46.98 |
| Summarization | news | 20 | -142.19 | 39.45 |
| Chat | curriculum_development | 4 | 55.51 | 88.88 |
| Chat | question_answering | 15 | 34.28 | 83.57 |
| Chat | data_analysis | 3 | 21.55 | 80.39 |
| Chat | character_creation | 12 | 16.62 | 79.16 |
| Chat | creative_write | 18 | 11.95 | 77.99 |
| Chat | guide_generation | 15 | 1.51 | 75.38 |
| Chat | idea_generation | 9 | -6.55 | 73.36 |
| Chat | script_write | 13 | -6.62 | 73.34 |
| Chat | question_generation | 10 | -19.50 | 70.13 |
| Chat | explanation | 2 | -23.04 | 69.24 |
| Chat | continue_write | 5 | -25.18 | 68.70 |
| Chat | academic_write | 14 | -28.75 | 67.81 |
| Chat | rewrite | 7 | -35.99 | 66.00 |
| Chat | report_write | 7 | -40.68 | 64.83 |
| Chat | science_problem_solve | 13 | -97.63 | 50.59 |
| Text completion | continuation | 32 | -28.87 | 67.78 |
| Text completion | imitative_writing | 23 | -37.31 | 65.67 |
| Text completion | style_transfer | 22 | -62.06 | 59.49 |
| Heuristic text generation | argumentative_writing | 23 | 1.27 | 75.32 |
| Heuristic text generation | story_writing | 25 | -2.16 | 74.46 |
| Heuristic text generation | roleplaying_writing | 25 | -6.07 | 73.48 |
| Heuristic text generation | keyword_writing | 25 | -12.72 | 71.82 |
| Heuristic text generation | screenplay_writing | 25 | -18.07 | 70.48 |

Small sub-categories (fewer than 5 items) are noisy.

### What the judge does well and badly

Read a sample of its reasons in `results/.../judgements/*.jsonl` (fields `flaws`, `reason`, `evaluation_score`). From spot checks:

- **Good catches:** a factual error ("quantum entanglement … has not yet been experimentally proven"), scored 0; a response of under 1,000 words when 5,000 were requested; a wrong percentage in a report summary.
- **Weaknesses:** it still sometimes invents meta-flaws ("does not explicitly state the requirement…"), occasionally marks story continuations down for introducing new characters despite being told not to, sets impossible standards ("does not eliminate *all* possible counterarguments"), and some reasons are generic ("lacks specificity").

## How this differs from the official HelloBench run

| | Official HelloBench | This repository |
|---|---|---|
| Model evaluated | Qwen2-7B-Instruct | Qwen2-7B-Instruct (same) |
| Generation | temperature 0.8, max 16,384 new tokens | temperature 0.8, **max 2,048 new tokens** (to fit Kaggle's 12-hour limit) |
| Judge | GPT-4o | **Qwen2.5-14B-Instruct, 4-bit NF4** (bitsandbytes) |
| Judge prompt | HelloBench prompt | **Strict prompt**: one-sentence "flaws" before every score, calibrated scale, no invented flaws, real word count given for length items (the official prompt is still available with `--judge_prompt official`) |
| Judge decoding | temperature 0.8, up to 3 tries | greedy (temperature 0), up to 3 tries (retries sample at 0.3) |
| Scoring | weights per sub-category, (x − 75) × 4, mean of 5 tasks | identical (weights copied from HelloBench `compute.py`; verified to reproduce the official 13.68 from the official judgements) |

Why not the official prompt with a small judge? In a test, Qwen2-7B judging its own answers with the official prompt gave **every** checklist item 1.0, so the scores carried no information. A stronger judge with a stricter prompt produces real, specific criticism.

## Repository layout

```
hellobench_qwen.py                      pipeline: generate → judge → score (CLI)
notebooks/
  hellobench_qwen_kaggle_t4x2.ipynb     full run on Kaggle GPU T4 x2 (what produced these results)
data/                                   the 5 HelloBench data files (647 items, with checklists)
results/qwen2_7b__judge_qwen25_14b/
  generations/                          Qwen2-7B-Instruct answers           {id, instruction, response}
  judgements/                           judge output per checklist item     + checklist_wise_evaluation
  scores/summary.json                   task and sub-category scores
  scores/per_item_scores.csv            one row per item
```

## Run it yourself

### On Kaggle (free GPU T4 × 2)
1. Create a notebook with **File → Import Notebook** → `notebooks/hellobench_qwen_kaggle_t4x2.ipynb`.
2. Settings: **Accelerator GPU T4 × 2**, **Internet on**. **Add Input → Upload** the 5 files from `data/` as a dataset.
3. Run once with `LIMIT = 2` (10 items) and read the judge's reasons (cell 6). Then set `LIMIT = 0` and use **Save Version → Save & Run All**.

The full run took about 10 hours: generation 3 h 50 min, judging 6 h 15 min. Judging would be faster with the current code (see the note below). If a run stops partway, start a new version with the stopped version's output added as an input (**Add Input → Your Work → Notebooks**), copy it into the working folder by adding `!cp -r /kaggle/input/<earlier-notebook>/runs /kaggle/working/` before the run cell, and run again: finished items are skipped.

### Command line (any CUDA GPU)
```bash
pip install -r requirements.txt
python hellobench_qwen.py generate --data_dir data --out_dir runs/my_run --gen_model Qwen/Qwen2-7B-Instruct
python hellobench_qwen.py judge    --data_dir data --out_dir runs/my_run --judge_model Qwen/Qwen2.5-14B-Instruct --judge_4bit --judge_prompt strict --judge_temperature 0
python hellobench_qwen.py score    --data_dir data --out_dir runs/my_run
```
Every stage resumes where it stopped. Run `python hellobench_qwen.py --help` for all options (batch sizes, memory budgets, official prompt, and so on). To rescore the included results: `python hellobench_qwen.py score --data_dir data --judge_dir results/qwen2_7b__judge_qwen25_14b/judgements --out_dir runs/rescore`.

**Note on the code version.** These results were produced by an earlier version of `hellobench_qwen.py`. In that version, one out-of-memory error during summarization judging permanently reduced the judge's batch size to 2 for the rest of the run, which made judging slower. The current version limits that cut to the affected group of items and gives the judge its own memory budget (`--judge_token_budget`). This changes speed only, not the scores.

## Credits and license

- Benchmark data, the official judge prompt and the checklist weights are from **HelloBench**: Que et al., *HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models* ([arXiv:2409.16191](https://arxiv.org/abs/2409.16191), [GitHub](https://github.com/Quehry/HelloBench)), MIT License. See `third_party/HelloBench_LICENSE`.
- Models: [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).
- Code in this repository: MIT License (see `LICENSE`).
