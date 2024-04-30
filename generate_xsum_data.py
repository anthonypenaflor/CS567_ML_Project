from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os
from data_generation_utils import sample_from_model, trim_to_shorter_length
import json

# For this one we may want to use non-instruction tuned models since we're not giving any instructions and we expect the models to complete the generation on their own
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # "mistralai/Mistral-7B-Instruct-v0.2"
DOWNLOAD_DIR = "/home/dafirebanks/projects/dont-stop-prompting/models"
BATCH_SIZE = 100
SEED = 42
XSUM_DIRPATH = "data/xsum"
N_DATA_SAMPLES = 5000

set_seed(SEED)

# Load data
if os.path.exists(XSUM_DIRPATH):
    print("XSUM Dataset found! Loading from path...")
    dataset = load_from_disk(XSUM_DIRPATH)
else:
    print(f"XSUM Dataset not found! Downloading and saving to {XSUM_DIRPATH}")
    dataset = load_dataset("EdinburghNLP/xsum")
    dataset.save_to_disk(XSUM_DIRPATH)

# Select a sample and generate prompts
dataset = dataset.shuffle(seed=SEED)
samples = dataset["train"].select(range(N_DATA_SAMPLES))
xsum_prompts = [sample["document"] for sample in samples]

# Load models
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=DOWNLOAD_DIR, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=DOWNLOAD_DIR)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Switch the padding side to the right - doesn't really matter because all texts should have at least 30 tokens, but this is to be consistent with the original code
if tokenizer.padding_side == "left":
    tokenizer.padding_side = "right"

sampling_kwargs = {
    "max_length": 200,
    "do_sample": True,
    "top_k": 40,
    "top_p": 0.96,
    "min_length": 150,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# Generate examples in batches so we don't run out of memory
data = {
    "original": [],
    "sampled": [],
}
for batch in range(len(xsum_prompts) // BATCH_SIZE):
    print("Generating samples for batch", batch, "of", len(xsum_prompts) // BATCH_SIZE)
    original_texts_batched = xsum_prompts[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
    sampled_text = sample_from_model(
        original_texts_batched,
        model,
        tokenizer,
        sampling_kwargs,
        min_words=55,
        n_prompt_tokens=30,
    )

    for o, s in zip(original_texts_batched, sampled_text):
        o, s = trim_to_shorter_length(o, s)
        data["original"].append(o)
        data["sampled"].append(s)

    # Store the generated essays in a jsonl file
    all_data = []
    for i in range(len(data["original"])):
        nongen = {"text": data["original"][i], "label": 0}
        gen = {"text": data["sampled"][i], "label": 1}

        all_data.append(nongen)
        all_data.append(gen)

    with open(f"data/xsum-gen-n={N_DATA_SAMPLES}.jsonl", "a") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
