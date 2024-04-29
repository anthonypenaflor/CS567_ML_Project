import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from data_generation_utils import sample_from_model
from collections import defaultdict

# For this task, we can use instruction tuned models if we give the model the original essay prompts, else we should use a non-instruction tuned model.
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

set2fullessays = defaultdict(list)

# Generate examples in batches so we don't run out of memory
for set, prompt in set2fullprompt.items():
    h_prompts = [prompt] * N_ESSAYS_PER_PROMPT
    for batch in range(len(h_prompts) // BATCH_SIZE):
        print("Generating samples for batch", batch, "of", len(h_prompts) // BATCH_SIZE)
        batch_prompts = h_prompts[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]

        sampled_texts = sample_from_model(
            batch_prompts, model, tokenizer, sampling_kwargs, min_words=55
        )
        set2fullessays[set].append(sampled_texts)

# Create a dataframe containing original and generated essays
print("Generated! Storing...")
df_full = pd.DataFrame(columns=["essay_set", "essay_id", "essay", "generated"])
for set, essays in set2fullessays.items():
    df_set = df[df["essay_set"] == set].copy().iloc[: len(essays)]
    # Add an extra column value of "0" for the original essays
    df_set["generated"] = 0

    # Add generated essays as additional rows
    for i, essay in enumerate(essays):
        row = {
            "essay_set": set,
            "essay_id": i,
            "essay": essay,
            "generated": 1,
            "prompt": set2fullprompt[set],
        }
        df_set = pd.concat([df_set, pd.DataFrame([row])], ignore_index=True)

    df_full = pd.concat([df_full, df_set], ignore_index=True)

train, test = train_test_split(df_full, test_size=0.2, random_state=42)

# Store
train.to_json(f"data/hewlett-n={N_ESSAYS_PER_PROMPT}-train.jsonl", orient="records", lines=True)
test.to_json(f"data/hewlett-n={N_ESSAYS_PER_PROMPT}-test.jsonl", orient="records", lines=True)
