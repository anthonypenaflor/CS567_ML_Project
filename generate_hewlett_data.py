from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from data_generation_utils import sample_from_model
from collections import defaultdict
import json

# For this task, we can use instruction tuned models if we give the model the original essay prompts, else we should use a non-instruction tuned model.
USE_ESSAY_INSTRUCTIONS = False
N_ESSAYS_PER_PROMPT = 1000
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DOWNLOAD_DIR = "/home/dafirebanks/projects/dont-stop-prompting/models"
CLEAN_DATA_FILEPATH = "data/hewlett-foundation-data/training_set_rel3_cleaned.csv"
SEED = 42
BATCH_SIZE = 50

ESSAYSET2PROMPT = {
    1: """More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.""",
    2: """Censorship in the Libraries
"All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us." --Katherine Paterson, Author
Write a persuasive essay to a newspaper reflecting your vies on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.""",
    7: """Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining.
Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.""",
    8: """We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.""",
}


set_seed(SEED)

# Load data
df = pd.read_csv(CLEAN_DATA_FILEPATH)

# For now, we'll only use the essays from essay_set 1, 2, 3, 4
df = df[df["essay_set"].isin(ESSAYSET2PROMPT.keys())]
df = df.reset_index(drop=True)
df["prompt"] = df["essay_set"].map(ESSAYSET2PROMPT)

# Generate prompts using the original essay instructions
h_ptemplate = """You will be provided with a prompt for an essay that needs to be written at the level of a student in 7-10th grade. You are an expert writer that knows how to write in different styles convincingly. You will read the prompt, and write an essay that is around 350 words.
Essay prompt: <ESSAY_PROMPT>
Essay:"""

set2fullprompt = {
    essay_set: h_ptemplate.replace("<ESSAY_PROMPT>", instruction)
    for essay_set, instruction in ESSAYSET2PROMPT.items()
}

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=DOWNLOAD_DIR, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=DOWNLOAD_DIR)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Switch the padding side to the right - doesn't really matter because all texts should have at least 30 tokens, but this is to be consistent with the original code
if tokenizer.padding_side == "left":
    tokenizer.padding_side = "right"

sampling_kwargs = {
    "max_length": 650,
    "do_sample": True,
    "top_k": 40,
    "top_p": 0.96,
    "min_length": 350,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

set2genessays = defaultdict(list)
set2humanessays = defaultdict(list)

# Generate examples in batches so we don't run out of memory
for essay_set, prompt in set2fullprompt.items():
    if USE_ESSAY_INSTRUCTIONS:
        h_prompts = [prompt] * N_ESSAYS_PER_PROMPT
    else:
        # Sample N_ESSAYS_PER_PROMPT essays from the dataset
        subset = df[df["essay_set"] == essay_set]
        h_prompts = subset.sample(min(N_ESSAYS_PER_PROMPT, len(subset)), random_state=SEED)[
            "essay"
        ].tolist()

    for batch in range(len(h_prompts) // BATCH_SIZE):
        print("Generating samples for batch", batch, "of", len(h_prompts) // BATCH_SIZE)
        batch_prompts = h_prompts[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]

        if USE_ESSAY_INSTRUCTIONS:  # Sample using the essay instructions
            sampled_texts = sample_from_model(
                batch_prompts, model, tokenizer, sampling_kwargs, min_words=55
            )
        else:  # Sample using the first n_prompt_tokens from each essay as the prompt
            sampled_texts = sample_from_model(
                batch_prompts, model, tokenizer, sampling_kwargs, min_words=55, n_prompt_tokens=30
            )
        set2genessays[essay_set].append(sampled_texts)
        set2humanessays[essay_set].append(batch_prompts)

    # Store the generated essays in a jsonl file
    all_data = []
    for i in range(len(set2genessays[essay_set])):
        nongen = {
            "essay": set2humanessays[essay_set][i],
            "label": 0,
            "prompt": set2fullprompt[essay_set],
            "essay_set": essay_set,
            "essay_id": i,
        }
        gen = {
            "essay": set2genessays[essay_set][i],
            "label": 1,
            "prompt": set2fullprompt[essay_set],
            "essay_set": essay_set,
            "essay_id": i,
        }

        all_data.append(nongen)
        all_data.append(gen)

    print("Generated! Storing...")
    with open(
        f"data/hewlett-n={N_ESSAYS_PER_PROMPT}-instruct={USE_ESSAY_INSTRUCTIONS}-essayset={essay_set}.jsonl",
        "a",
    ) as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
