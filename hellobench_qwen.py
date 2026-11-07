#!/usr/bin/env python3
"""
HelloBench with Qwen 7B: generate -> judge (Qwen as the judge) -> score.

Follows the official HelloBench code (github.com/Quehry/HelloBench, MIT License):
  * generation : run.py / run_qwen2_7b   -> user message = instruction, sampling, temperature 0.8
  * judging    : llm_judge.py            -> the official checklist-wise judge prompt (verbatim), plus a
                                              stricter prompt that is used by default (see below)
  * scoring    : results/scripts/compute.py -> per-sub-category checklist weights (WEIGHT_DICT),
                                              rescale (score - 75) * 4, overall = mean of 5 categories

Differences from the official run (needed for a free Kaggle T4 x2 GPU):
  * the judge is a Qwen model, not GPT-4o; by default with a stricter prompt (--judge_prompt strict),
    because small judges given the official prompt score almost everything 1.0
    (use --judge_prompt official for the verbatim HelloBench prompt)
  * max_new_tokens defaults to 2048 for generation (official: 16384) and 1024 for judging (official: 4096)

Files
-----
  data/*.jsonl  --generate-->  generations/<task>_results.jsonl   {"id","instruction","response"}
                --judge----->  judgements/<task>_results.jsonl    + "checklist_wise_evaluation"
                --score----->  scores/summary.json, scores/per_item_scores.csv

Usage
-----
  python hellobench_qwen.py all      --data_dir data --out_dir runs/qwen2_7b
  python hellobench_qwen.py generate --data_dir data --out_dir runs/qwen2_7b
  python hellobench_qwen.py judge    --data_dir data --out_dir runs/qwen2_7b
  python hellobench_qwen.py score    --data_dir data --out_dir runs/qwen2_7b
  # score any judgement folder, e.g. the official GPT-4o judgements:
  python hellobench_qwen.py score --data_dir data --judge_dir official_judgements --out_dir runs/official

Data files may be named "<task>.jsonl" or "<prefix>-<task>.jsonl".
Needs torch, transformers, accelerate (preinstalled on Kaggle). Re-running resumes.
"""

import argparse
import ast
import csv
import gc
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

TASKS = ["open_ended_qa", "summarization", "chat", "text_completion", "heuristic_text_generation"]
DEFAULT_MODEL = "Qwen/Qwen2-7B-Instruct"   # the "Qwen-2-7B" model evaluated in HelloBench

# Official checklist weights per sub-category, from HelloBench results/scripts/compute.py
# (each list sums to 100; one weight per checklist item, in order).
WEIGHT_DICT = {
    'travel': [14.816357060044538, 9.912640189521913, 6.854178078417421, 16.548365732493735, 12.49440306294194, 19.925026350726633, 19.449029525853824],
    'Tech': [9.730382391494699, 15.439961810101806, 8.71267868266836, 17.047912114497525, 8.188210881912578, 18.27285649160541, 22.607997627719627],
    'Sport': [10.470669731543392, 9.628138754444748, 5.8376755613192275, 18.908737698203687, 11.170106247242, 22.555525595175727, 21.42914641207122],
    'Science': [10.215624094265426, 11.85130160404758, 13.199743482703303, 15.771351181725294, 10.772433227719386, 18.259334358981764, 19.93021205055725],
    'music': [10.570558131445923, 10.250703197641212, 8.71555097518865, 20.767746121844873, 15.130089494653312, 17.459999261696932, 17.10535281752909],
    'health': [14.409021815474166, 8.996654196952731, 9.058311451032425, 20.374818020413127, 13.113089390107218, 13.622853268531996, 20.425251857488348],
    'write': [17.20178646947119, 11.647858398657238, 13.549614784591249, 18.451414657788348, 8.415665936780018, 15.693785853424465, 15.039873899287489],
    'book': [10.987786546263385, 10.80583601777249, 11.110641533898052, 21.917965372650762, 7.7575931269958955, 15.37978249492496, 22.040394907494466],
    'food': [10.88637513076461, 11.972608253231327, 13.762365658958538, 18.449103701644535, 10.828866753473488, 15.403319360473219, 18.69736114145427],
    'movie': [13.987702750429126, 13.781107282170971, 10.70081300442185, 14.950249677014197, 9.043151114164273, 14.990326778304123, 22.54664939349545],
    'long_dialogue': [12.655129633685263, 12.128629670452108, 15.417359033606798, 8.805077038076321, 22.44683162734655, 19.3826287336546, 9.164344263178364],
    'blogs': [7.586054691386359, 19.535003668901773, 15.361732493611802, 17.16924394200404, 16.86984484117092, 23.47812036292512],
    'academic_article': [7.1513786821899865, 13.027210863148744, 17.517148962264663, 14.222879878391684, 18.018026707391165, 30.06335490661375],
    'report': [8.962021075489186, 17.645150656180856, 17.07695284575253, 12.962529199816222, 18.77731391007885, 24.57603231268235],
    'news': [5.746318852823619, 18.828108458188307, 18.348616241165825, 16.546667215885762, 20.49878321641544, 20.03150601552105],
    'question_generation': [15.630644520221793, 17.815836405315725, 5.260151108793491, 5.260151108793491, 30.281435872156237, 25.751780984719254],
    'character_creation': [13.387472615551518, 16.154170714995903, 5.3564749039425825, 17.745872651899493, 27.8316766814783, 19.5243324321322],
    'script_write': [16.020800876858075, 12.537284513149297, 7.583604904411543, 10.962130120971509, 21.55253807214911, 31.343641512460472],
    'report_write': [23.715406207770044, 11.322739017895511, 6.455129156251138, 7.266817046605194, 20.795517896089837, 30.444390675388284],
    'science_problem_solve': [14.532002727010074, 13.988091295206875, 5.78110629330191, 13.906652976851941, 29.749526456076786, 22.042620251552407],
    'academic_write': [18.274980968685334, 17.668799475735167, 5.373737221539396, 15.33990358340595, 27.116855004727352, 16.225723745906805],
    'guide_generation': [24.991645087603484, 12.995989180532902, 11.348066943331492, 13.176536571757417, 19.238518079064633, 18.249244137710075],
    'creative_write': [20.56735945510573, 13.865892755893375, 9.95947810767433, 16.610586533096885, 21.307725530193018, 17.68895761803666],
    'question_answering': [14.257396776453227, 12.59853746572811, 5.7410180060529985, 15.959901439015228, 28.83810948056622, 22.60503683218423],
    'curriculum_development': [20.68850512855878, 22.200461872620195, 5.8343282109082, 5.8343282109082, 17.89639729448703, 27.545979282517592],
    'continue_write': [18.669885223104068, 21.418933575454858, 13.841889274353397, 6.502715042824038, 17.14288545529491, 22.423691428968727],
    'idea_generation': [16.608491609592104, 24.45709647197801, 12.235414254617053, 5.504078770891624, 18.79437075684626, 22.400548136074956],
    'data_analysis': [18.29675276651988, 5.722157365550123, 5.740218388378298, 20.92508664739828, 26.510684489335194, 22.80510034281823],
    'rewrite': [20.801683025093183, 8.510828270810512, 11.130570080160155, 13.722027611417639, 19.803701313664753, 26.03118969885375],
    'explanation': [10.313604819556165, 18.631545950717513, 16.412914400566404, 11.838586893660816, 19.111282531748692, 23.69206540375043],
    'continuation': [21.427707308340644, 19.022158840412466, 16.220256947514333, 20.57043807105919, 22.759438832673375],
    'imitative_writing': [19.87078310837695, 19.793380163686955, 19.346176082395687, 21.77086167116268, 19.218798974377737],
    'style_transfer': [16.438886068023052, 18.226961726018953, 21.448441756584106, 23.961762450033103, 19.923947999340776],
    'story_writing': [23.5319284319259, 22.420937450120597, 10.539906363853124, 17.047083302574496, 26.460144451525895],
    'keyword_writing': [16.27370693012242, 27.30111800645728, 15.728682122621054, 18.81389796206547, 21.882594978733778],
    'screenplay_writing': [19.822086987393824, 20.973270981524056, 17.095645893112255, 19.56592278203641, 22.543073355933444],
    'argumentative_writing': [18.302865025230115, 24.50501277580138, 20.483643154138807, 14.552018259438853, 22.15646078539085],
    'roleplaying_writing': [18.23837535323756, 22.299189217994243, 12.860964861892231, 19.918295740192793, 26.683174826683164],
}

VALID_SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_file(directory, task, suffix=".jsonl"):
    """Find '<task><suffix>' or '<prefix>-<task><suffix>' in directory."""
    exact = os.path.join(directory, f"{task}{suffix}")
    if os.path.exists(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(directory, f"*-{task}{suffix}")))
    if len(hits) > 1:
        log(f"  [warn] several files match {task}{suffix} in {directory}: {hits}; using {hits[0]}")
    return hits[0] if hits else None


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    return {r["id"] for r in read_jsonl(path)}


# --------------------------------------------------------------------------- #
# Model wrapper: Hugging Face transformers, batched
# --------------------------------------------------------------------------- #
class ChatModel:
    def __init__(self, model_name, dtype="auto", max_model_len=32768,
                 max_batch=16, token_budget=48000, gpu_mem_gib=None, load_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.max_batch = max_batch
        self.token_budget = token_budget

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
                torch_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                torch_dtype = torch.float16          # T4 / P100: no bfloat16
            else:
                torch_dtype = torch.float32
        else:
            torch_dtype = {"half": torch.float16, "float16": torch.float16,
                           "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in transformers 4.56
        kwargs = {dtype_key: torch_dtype, "attn_implementation": "sdpa"}
        from transformers import AutoConfig
        already_quantized = getattr(AutoConfig.from_pretrained(model_name), "quantization_config", None)
        if load_4bit and not already_quantized:  # 4-bit NF4 (bitsandbytes): 14B needs ~10 GB instead of ~29 GB
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True)
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if gpu_mem_gib:  # cap weights per GPU so layers are split evenly, leaving room for batches
                kwargs["max_memory"] = {i: f"{gpu_mem_gib}GiB" for i in range(torch.cuda.device_count())}
        log(f"loading {model_name} ({torch_dtype}{', 4-bit' if load_4bit else ''}) ...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        if hasattr(self.model, "hf_device_map"):
            devs = defaultdict(int)
            for d in self.model.hf_device_map.values():
                devs[str(d)] += 1
            log(f"modules per device: {dict(devs)}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                log(f"GPU{i} memory used after load: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")

    def _prompt(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _n_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def fit_prompt(self, messages, max_new_tokens):
        """Return (prompt, max_new_tokens) that fit in max_model_len."""
        prompt = self._prompt(messages)
        n = self._n_tokens(prompt)
        if n + max_new_tokens + 16 <= self.max_model_len:
            return prompt, max_new_tokens
        new_max = self.max_model_len - n - 16
        if new_max >= 1024:
            return prompt, new_max
        content = messages[-1]["content"]
        ids = self.tokenizer(content, add_special_tokens=False)["input_ids"]
        overflow = n - (self.max_model_len - 1024 - 16)
        keep = max(0, len(ids) - overflow - 32)
        cut = (self.tokenizer.decode(ids[: keep // 2]) + "\n...[truncated]...\n"
               + self.tokenizer.decode(ids[len(ids) - keep // 2:]))
        log(f"  [warn] prompt truncated ({n} tokens)")
        return self._prompt(messages[:-1] + [{"role": "user", "content": cut}]), 1024

    def _generate_batch(self, prompts, max_new_tokens, temperature):
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self.input_device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)
        if temperature and temperature > 0:
            # like the official run: only temperature is set; top_p/top_k/repetition_penalty
            # come from the model's own generation_config
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)
        texts = self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        del enc, out
        return texts

    def generate(self, list_of_messages, max_new_tokens=2048, temperature=0.8):
        torch = self.torch
        fitted = [self.fit_prompt(m, max_new_tokens) for m in list_of_messages]
        lens = [self._n_tokens(p) for p, _ in fitted]
        order = sorted(range(len(fitted)), key=lambda i: -lens[i])  # longest first
        results = [None] * len(fitted)
        max_batch = self.max_batch   # an out-of-memory cut only applies to this call, not the rest of the run

        pos = 0
        while pos < len(order):
            batch = [order[pos]]
            longest = lens[order[pos]]
            mnt = fitted[order[pos]][1]
            while pos + len(batch) < len(order) and len(batch) < max_batch:
                nxt = order[pos + len(batch)]
                cand_mnt = min(mnt, fitted[nxt][1])
                if (len(batch) + 1) * (longest + cand_mnt) > self.token_budget:
                    break
                batch.append(nxt)
                mnt = cand_mnt

            prompts = [fitted[i][0] for i in batch]
            t0 = time.time()
            try:
                texts = self._generate_batch(prompts, mnt, temperature)
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if len(batch) == 1:
                    log(f"  [warn] out of memory on one prompt ({longest} tokens); retrying with 1024 new tokens")
                    texts = self._generate_batch(prompts, min(mnt, 1024), temperature)
                else:
                    max_batch = max(1, len(batch) // 2)
                    log(f"  [warn] out of GPU memory with batch {len(batch)}; "
                        f"retrying with batches of up to {max_batch} for this group of items")
                    continue
            for i, t in zip(batch, texts):
                results[i] = t
            pos += len(batch)
            log(f"  batch of {len(batch)} (prompt <= {longest} tokens) done in "
                f"{time.time() - t0:.0f}s  [{pos}/{len(order)}]")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results


# --------------------------------------------------------------------------- #
# Stage 1: generation (official: user message = instruction, temperature 0.8)
# --------------------------------------------------------------------------- #
def run_generate(model, args):
    gen_dir = os.path.join(args.out_dir, "generations")
    os.makedirs(gen_dir, exist_ok=True)

    for task in args.tasks:
        data_path = find_file(args.data_dir, task)
        if not data_path:
            log(f"[generate] no data file for {task}, skipping")
            continue
        out_path = os.path.join(gen_dir, f"{task}_results.jsonl")
        done = load_done_ids(out_path)
        items = [r for r in read_jsonl(data_path) if r["id"] not in done]
        if args.limit:
            items = items[: max(0, args.limit - len(done))]
        log(f"[generate] {task}: {len(items)} to do ({len(done)} already done)")

        for i in range(0, len(items), args.chunk_size):
            chunk = items[i: i + args.chunk_size]
            msgs = [[{"role": "user", "content": r["instruction"]}] for r in chunk]
            outs = model.generate(msgs, max_new_tokens=args.gen_max_tokens,
                                  temperature=args.gen_temperature)
            append_jsonl(out_path, [{"id": r["id"], "instruction": r["instruction"], "response": o}
                                    for r, o in zip(chunk, outs)])
            log(f"[generate] {task}: saved {len(done) + i + len(chunk)} items")


# --------------------------------------------------------------------------- #
# Stage 2: checklist-wise judging (official HelloBench prompt verbatim, plus the strict prompt used by default)
# --------------------------------------------------------------------------- #
SYS_PROMPT = """You are a helpful evaluator. Your task is to evaluate the checklists of the responses given by the Large Language Models (LLMs) based on user instructions. These checklists consist of yes or no questions."""
USER_PROMPT = """Your core task is to evaluate the checklists based on the user's instruction and LLM's response, with each checklist item being a yes or no question indicating a specific aspect that the LLM's response should meet. You need to judge the checklist item based on the instruction and response. The evaluation results are scored from 0 to 1, with 5 scores in total, which are:

0: The response fails to meet the checklist requirements, demonstrating substantial need for improvement across multiple areas.
0.25: The response partially meets some checklist requirements, but significant elements remain unaddressed.
0.5: The response meets several checklist requirements, yet the overall evaluation appears ambiguous or unclear.
0.75: The response aligns with most checklist requirements, though there are still minor areas that could be refined or enhanced.
1: The response fully satisfies all checklist requirements, with no identifiable issues or areas for improvement. It means this response is already perfect; you can't find any significant flaws in it.

Here is the instruction:
{{\"instruction\": {instruction}}}

Here is the response given by LLM:
{{\"response\": {response}}}

Since the response may be rather long, I am specifically reminding you here that the response has ended.

Here are checklists of this instruction:
{{\"checklists\": {checklists}}}

To further remind you, I will repeat my requirements:

Your core task is to evaluate the checklists based on the user's instruction and LLM's response, with each checklist item being a yes or no question indicating a specific aspect that the LLM's response should meet. You need to judge the checklist item based on the instruction and response. The evaluation results are scored from 0 to 1, with 5 scores in total, which are:

0: The response fails to meet the checklist requirements, demonstrating substantial need for improvement across multiple areas.
0.25: The response partially meets some checklist requirements, but significant elements remain unaddressed.
0.5: The response meets several checklist requirements, yet the overall evaluation appears ambiguous or unclear.
0.75: The response aligns with most checklist requirements, though there are still minor areas that could be refined or enhanced.
1: The response fully satisfies all checklist requirements, with no identifiable issues or areas for improvement. It means this response is already perfect; you can't find any significant flaws in it.

Always provide the reason for your evaluation results. You should be strict but fair in your evaluation. A score of 1 means that the response perfectly meets all the checklist requirements and you think there are really no room for improvements. When giving a score of 1, you need to carefully consider whether this checklist has been perfectly satisfied.

Evaluate all the checklists and return the evaluation results of the checklists. Output a Python List consisting of the Python Dictionary formatted as follows:
[{{\"checklist_id\": \"the id of the checklist\", \"reason\": \"The reason for your evaluation results\", \"evaluation_score\": \"Your evaluation score for this checklist\"}},{{\"checklist_id\": \"the id of the checklist\", \"reason\": \"The reason for your evaluation results\", \"evaluation_score\": \"Your evaluation score for this checklist\"}}]

There are total {num_checklist} checklists that you need to evaluate. The length of the output list is equal to the number of checklists and you should give an evaluation score for each checklist. You should be strict to the evaluation to further compare the responses from different models. Your response must be a valid Python List and should contain nothing else, as it will be directly executed in Python."""


STRICT_SYS_PROMPT = """You are a strict, critical expert evaluator of long-form text written by AI models. Your scores are used to rank models, so lenient scoring makes them useless. You look for concrete weaknesses before deciding any score."""
STRICT_USER_PROMPT = """Evaluate the LLM response below against each checklist item. Each checklist item is a yes/no question about one aspect the response should meet.

Here is the instruction:
<instruction>
{instruction}
</instruction>

Here is the response given by the LLM (it contains about {n_words} words):
<response>
{response}
</response>

The response has ended.

Here are the checklists of this instruction:
{checklists}

Scores (pick exactly one per checklist item):
0: fails the item; substantial problems.
0.25: partially meets it, but significant elements are missing or wrong.
0.5: meets it only in part; clear weaknesses.
0.75: meets most of it, with minor but real issues.
1: perfect for this item; you searched and found no flaw at all.

Rules you must follow:
1. For EACH item, first write "flaws": the most important specific shortcoming(s) you found, in ONE short sentence (max 30 words), pointing to or quoting the response. Look hard: missing details, vague or generic content, repetition, padding, factual slips, unexplained terms, weak structure, clichés, ignored parts of the instruction, length that does not match what was asked. Write "none" only if you truly found nothing.
2. A flaw must be something actually wrong with or missing from the response for THIS checklist item. Do not invent flaws, and do not penalise the response for not restating the instruction or the checklist. Content that the instruction asks for is not a flaw: for example, a story continuation is supposed to add new events, characters and ideas; judge whether they fit the original story, not whether they are new.
3. Then write "reason": ONE short sentence (max 30 words) about specific content of the response. Do not restate the checklist question or repeat "flaws".
4. Then give "evaluation_score". A score of 1 is only allowed when "flaws" is "none", and that should be rare. A typical competent response deserves 0.5 or 0.75 on most items.
5. If a checklist item says that any single error means it is not satisfied, apply that literally.
6. For length-related items, use the word count given above, not your impression.

Output ONLY a JSON list with {num_checklist} objects, one per checklist item, in order, and nothing else:
[{{"checklist_id": 0, "flaws": "...", "reason": "...", "evaluation_score": 0.75}}, ...]"""


def build_judge_messages(item, response, prompt_style="official"):
    # official: responses longer than 15000 words are cut to 16000 words (sic, copied as is from llm_judge.py)
    words = response.split()
    if len(words) > 15000:
        response = " ".join(words[:16000])
    checklists = item["checklists"]
    checklist_list = [{"checklist_id": i, "checklist_content": ck} for i, ck in enumerate(checklists)]
    if prompt_style == "strict":
        user = STRICT_USER_PROMPT.format(
            instruction=item["instruction"], response=response, n_words=len(response.split()),
            checklists="\n".join(f"[{i}] {ck}" for i, ck in enumerate(checklists)),
            num_checklist=len(checklists))
        return [{"role": "system", "content": STRICT_SYS_PROMPT}, {"role": "user", "content": user}]
    user = USER_PROMPT.format(instruction=item["instruction"], response=response,
                              checklists=json.dumps(checklist_list, ensure_ascii=False),
                              num_checklist=len(checklists))
    return [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}]


def to_score(value):
    """Turn the judge's evaluation_score into one of 0, .25, .5, .75, 1 (None if unusable)."""
    try:
        v = float(str(value).strip().strip("'\""))
    except (TypeError, ValueError):
        return None
    if v < 0 or v > 1:
        return None
    return min(VALID_SCORES, key=lambda s: abs(s - v))


def parse_judgement(text, n_checklists):
    """Parse the judge output into [{checklist_id, reason, evaluation_score}] or None."""
    t = text.replace("```json", "").replace("```python", "").replace("```", "").strip()
    m = re.search(r"\[.*\]", t, flags=re.S)
    parsed = None
    if m:
        block = m.group(0)
        for loader in (json.loads, ast.literal_eval,
                       lambda s: json.loads(s.replace("\n", " ").replace("\\", ""))):
            try:
                parsed = loader(block)
                break
            except Exception:
                continue

    results = {}
    if isinstance(parsed, list):
        for pos, e in enumerate(parsed):
            if not isinstance(e, dict):
                continue
            try:
                cid = int(e.get("checklist_id", pos))
            except (TypeError, ValueError):
                cid = pos
            score = to_score(e.get("evaluation_score"))
            if score is not None:
                results[cid] = {"checklist_id": cid, "reason": str(e.get("reason", "")).strip(),
                                "evaluation_score": score}
                if "flaws" in e:
                    results[cid]["flaws"] = str(e["flaws"]).strip()
    else:  # last resort: pull the fields out with a regex
        for cid, sc in re.findall(
                r"""["']checklist_id["']\s*:\s*["']?(\d+)["']?.*?["']evaluation_score["']\s*:\s*["']?([0-9.]+)""",
                t, flags=re.S):
            score = to_score(sc)
            if score is not None:
                results[int(cid)] = {"checklist_id": int(cid), "reason": "", "evaluation_score": score}

    if all(i in results for i in range(n_checklists)):
        return [results[i] for i in range(n_checklists)]
    return None


def run_judge(model, args):
    gen_dir = args.gen_dir or os.path.join(args.out_dir, "generations")
    judge_dir = os.path.join(args.out_dir, "judgements")
    os.makedirs(judge_dir, exist_ok=True)

    for task in args.tasks:
        data_path = find_file(args.data_dir, task)
        gen_path = find_file(gen_dir, task, "_results.jsonl")
        if not data_path or not gen_path:
            log(f"[judge] missing data or generation file for {task}, skipping")
            continue
        data = {r["id"]: r for r in read_jsonl(data_path)}
        out_path = os.path.join(judge_dir, f"{task}_results.jsonl")
        done = load_done_ids(out_path)
        gens = [g for g in read_jsonl(gen_path) if g["id"] not in done and g["id"] in data]
        if args.limit:
            gens = gens[: max(0, args.limit - len(done))]
        log(f"[judge] {task}: {len(gens)} to do ({len(done)} already done)")

        for i in range(0, len(gens), args.chunk_size):
            chunk = gens[i: i + args.chunk_size]
            pending = list(range(len(chunk)))
            parsed = [None] * len(chunk)
            raw_out = [""] * len(chunk)

            for attempt in range(args.judge_tries):  # official: up to 3 tries
                if not pending:
                    break
                msgs = [build_judge_messages(data[chunk[k]["id"]], chunk[k]["response"], args.judge_prompt)
                        for k in pending]
                # retries sample a little, so a greedy judge does not repeat the same broken output
                temp = args.judge_temperature if attempt == 0 else max(args.judge_temperature, 0.3)
                outs = model.generate(msgs, max_new_tokens=args.judge_max_tokens, temperature=temp)
                still = []
                for k, o in zip(pending, outs):
                    raw_out[k] = o
                    parsed[k] = parse_judgement(o, len(data[chunk[k]["id"]]["checklists"]))
                    if parsed[k] is None:
                        still.append(k)
                if still and attempt + 1 < args.judge_tries:
                    log(f"  {len(still)} judgement(s) could not be parsed, trying again")
                pending = still

            rows = []
            for k, g in enumerate(chunk):
                row = {"id": g["id"], "instruction": g["instruction"], "response": g["response"],
                       "checklist_wise_evaluation": parsed[k] or []}
                if parsed[k] is None:  # official behaviour: empty list -> item left out of the score
                    log(f"  [warn] judge output for {g['id']} unusable after {args.judge_tries} tries; "
                        f"it will be left out of the score")
                    row["raw_judgement"] = raw_out[k]
                rows.append(row)
            append_jsonl(out_path, rows)
            log(f"[judge] {task}: saved {len(done) + i + len(chunk)} items")


# --------------------------------------------------------------------------- #
# Stage 3: scoring (official compute.py)
# --------------------------------------------------------------------------- #
def count_words(text):
    try:
        import nltk
        return len(nltk.word_tokenize(text))   # official length measure
    except Exception:
        return len(re.findall(r"\w+|[^\w\s]", text))


def run_score(args):
    judge_dir = args.judge_dir or os.path.join(args.out_dir, "judgements")
    score_dir = os.path.join(args.out_dir, "scores")
    os.makedirs(score_dir, exist_ok=True)

    per_item, summary = [], {"tasks": {}}
    for task in args.tasks:
        jpath = find_file(judge_dir, task, "_results.jsonl")
        dpath = find_file(args.data_dir, task)
        if not jpath or not dpath:
            log(f"[score] missing judgement or data file for {task}, skipping")
            continue
        id_cat = {r["id"]: r["category"] for r in read_jsonl(dpath)}

        seen, skipped = set(), 0
        scores, lengths, sub = [], [], defaultdict(list)
        for r in read_jsonl(jpath):
            _id = r["id"]
            if _id not in id_cat or _id in seen:
                continue
            seen.add(_id)
            weights = WEIGHT_DICT[id_cat[_id]]
            ev = r.get("checklist_wise_evaluation") or []
            try:
                vals = [float(e["evaluation_score"]) for e in sorted(ev, key=lambda e: int(e["checklist_id"]))]
                ok = len(vals) == len(weights) and all(0 <= v <= 1 for v in vals)
            except (KeyError, TypeError, ValueError):
                ok = False
            if not ok:
                skipped += 1
                continue
            s = sum(v * w for v, w in zip(vals, weights))        # 0-100
            n_words = count_words(r["response"])
            scores.append(s); lengths.append(n_words); sub[id_cat[_id]].append(s)
            per_item.append({"task": task, "sub_category": id_cat[_id], "id": _id,
                             "weighted_score": round(s, 2), "response_words": n_words,
                             "checklist_scores": " ".join(f"{v:g}" for v in vals)})

        if not scores:
            log(f"[score] {task}: no usable judgements")
            continue
        raw = sum(scores) / len(scores)
        summary["tasks"][task] = {
            "n_scored": len(scores), "n_skipped": skipped,
            "score": round((raw - 75) * 4, 2),         # official rescaled score
            "raw_weighted_score": round(raw, 2),       # 0-100 before rescaling
            "avg_length_words": round(sum(lengths) / len(lengths)),
            "sub_categories": {k: {"n": len(v), "raw_weighted_score": round(sum(v) / len(v), 2),
                                   "score": round((sum(v) / len(v) - 75) * 4, 2)}
                               for k, v in sorted(sub.items())},
        }

    t = summary["tasks"]
    if t:
        summary["overall_score"] = round(sum(v["score"] for v in t.values()) / len(t), 2)
        summary["overall_raw_weighted_score"] = round(sum(v["raw_weighted_score"] for v in t.values()) / len(t), 2)
        summary["overall_avg_length_words"] = round(sum(v["avg_length_words"] for v in t.values()) / len(t))
        if len(t) < len(TASKS):
            summary["note"] = f"only {len(t)} of {len(TASKS)} tasks scored; overall is not comparable to the paper"

    with open(os.path.join(score_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(score_dir, "per_item_scores.csv"), "w", newline="", encoding="utf-8") as f:
        if per_item:
            w = csv.DictWriter(f, fieldnames=list(per_item[0].keys()))
            w.writeheader(); w.writerows(per_item)

    if not t:
        log("[score] nothing to score")
        return summary
    print("\n" + "=" * 78)
    print(f"{'task':28s}{'scored':>8s}{'skipped':>9s}{'score':>9s}{'raw 0-100':>12s}{'avg words':>11s}")
    print("-" * 78)
    for task, v in t.items():
        print(f"{task:28s}{v['n_scored']:>8d}{v['n_skipped']:>9d}{v['score']:>9.2f}"
              f"{v['raw_weighted_score']:>12.2f}{v['avg_length_words']:>11d}")
        for name, s in v["sub_categories"].items():
            print(f"    {name:24s}{s['n']:>8d}{'':>9s}{s['score']:>9.2f}{s['raw_weighted_score']:>12.2f}")
    print("-" * 78)
    print(f"{'OVERALL (mean of tasks)':28s}{'':>8s}{'':>9s}{summary['overall_score']:>9.2f}"
          f"{summary['overall_raw_weighted_score']:>12.2f}{summary['overall_avg_length_words']:>11d}")
    print("=" * 78)
    print("score = (weighted checklist score - 75) * 4, as in HelloBench. 100 is perfect; 0 = all 'mostly met' (0.75).")
    print(f"written: {score_dir}/summary.json, {score_dir}/per_item_scores.csv")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["generate", "judge", "score", "all"])
    p.add_argument("--data_dir", default="data", help="folder with the 5 HelloBench data .jsonl files")
    p.add_argument("--out_dir", default="runs/qwen2_7b")
    p.add_argument("--gen_dir", default=None, help="judge generations from this folder instead of out_dir")
    p.add_argument("--judge_dir", default=None, help="score judgements from this folder instead of out_dir")
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--limit", type=int, default=0, help="only the first N items per task (for testing)")

    p.add_argument("--gen_model", default=DEFAULT_MODEL)
    p.add_argument("--judge_model", default=None, help="defaults to --gen_model")
    p.add_argument("--dtype", default="auto", help="auto | half | bfloat16 | float32 (auto = fp16 on T4)")
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--gpu_mem_gib", type=float, default=None,
                   help="max GiB of weights per GPU (9 on T4 x2 splits the model evenly)")
    p.add_argument("--max_batch", type=int, default=16, help="max prompts generated together")
    p.add_argument("--token_budget", type=int, default=48000,
                   help="max batch_size x (longest prompt + new tokens) per batch")
    p.add_argument("--chunk_size", type=int, default=32, help="items saved to disk at a time")

    p.add_argument("--gen_max_tokens", type=int, default=2048, help="official HelloBench: 16384")
    p.add_argument("--gen_temperature", type=float, default=0.8, help="official HelloBench: 0.8")
    p.add_argument("--judge_max_tokens", type=int, default=1024, help="official HelloBench: 4096")
    p.add_argument("--judge_temperature", type=float, default=0.0, help="official HelloBench: 0.8; 0 = greedy")
    p.add_argument("--judge_tries", type=int, default=3, help="official HelloBench: 3")
    p.add_argument("--judge_prompt", choices=["strict", "official"], default="strict",
                   help="strict = asks for concrete flaws before each score and calibrates the scale; "
                        "official = HelloBench's prompt verbatim (small judges give almost everything 1.0)")
    p.add_argument("--judge_4bit", action="store_true", help="load the judge in 4-bit (needs bitsandbytes)")
    p.add_argument("--judge_gpu_mem_gib", type=float, default=None, help="like --gpu_mem_gib, for the judge")
    p.add_argument("--judge_token_budget", type=int, default=None,
                   help="token budget for the judge (a 14B model needs ~3x more memory per token than a 7B)")
    p.add_argument("--free_gen_cache", action="store_true",
                   help="after generation, delete the generation model's downloaded files to free disk for the judge")
    args = p.parse_args()
    args.judge_model = args.judge_model or args.gen_model

    def load(name, judge=False):
        budget = (args.judge_token_budget or args.token_budget) if judge else args.token_budget
        return ChatModel(name, args.dtype, args.max_model_len, args.max_batch, budget,
                         (args.judge_gpu_mem_gib or args.gpu_mem_gib) if judge else args.gpu_mem_gib,
                         load_4bit=judge and args.judge_4bit)

    model = None
    if args.stage in ("generate", "all"):
        model = load(args.gen_model)
        run_generate(model, args)

    if args.stage in ("judge", "all"):
        if model is None or model.model_name != args.judge_model or args.judge_4bit:
            if model is not None:
                del model
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                if args.free_gen_cache and os.path.isdir(args.gen_model) is False:
                    import shutil
                    from huggingface_hub import constants
                    d = os.path.join(constants.HF_HUB_CACHE, "models--" + args.gen_model.replace("/", "--"))
                    if os.path.isdir(d):
                        shutil.rmtree(d, ignore_errors=True)
                        log(f"deleted downloaded files of {args.gen_model} to free disk space")
            model = load(args.judge_model, judge=True)
        run_judge(model, args)

    if args.stage in ("score", "all"):
        run_score(args)


if __name__ == "__main__":
    main()
