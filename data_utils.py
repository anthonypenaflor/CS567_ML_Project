from llm_inference_utils import InferenceMode
import json
from transformers import AutoTokenizer
import os
import pandas as pd

LABEL_TOKENS_DIRPATH = "data/label_tokens"
DATA_CONFIGS_DIRPATH = "data/configs"


def format_one_example_for_inference(d: dict, prompt_template: str, label=None):
    return prompt_template.replace("<TEXT>", d["text"])


def format_one_labeled_example(d: dict, prompt_template: str):
    return prompt_template.replace("<TEXT>", d["text"]).replace("<LABEL>", d["label"])


def generate_inference_examples(
    test_data: list[dict],
    prompt_template: str,
) -> list[str]:
    """Generate prompts for inference
    Args:
        data: list[dict]
            The data to generate prompts for
        inference_mode: InferenceMode
            The inference mode to use
    """
    return [format_one_example_for_inference(item, prompt_template) for item in test_data]


def load_demontration_prompt_template(data_config_filepath: str) -> str:
    """Load the demonstration prompt template from the data config file
    Args:
        data_config_filepath: str
            The path to the data config JSON file
    Returns:
        str
            The demonstration prompt template
    """
    with open(data_config_filepath, "r") as f:
        config = json.load(f)
    return config["demonstration_prompt"]


def load_inference_prompt_template(data_config_filepath: str) -> str:
    """Load the inference prompt template from the data config file
    Args:
        data_config_filepath: str
            The path to the data config JSON file
    Returns:
        str
            The inference prompt template
    """
    with open(data_config_filepath, "r") as f:
        config = json.load(f)
    return config["inference_prompt"]


def format_prefix(labeled_data: list[dict], task_instruction: str, prompt_template: str) -> str:
    """Format the prefix for the prompt. The prefix is a combination of unlabeled and labeled data, where the unlabeled data goes first (if any) and then we have labeled examples separated by a newline.
    Args:
        labeled_data: list[dict]
            The labeled data
        prompt_template: str
            The prompt template for labeled examples
    Returns:
        str
            The formatted prefix
    """
    if not labeled_data:  # For zero-shot learning
        return ""

    prefix = [task_instruction]

    for item in labeled_data:
        prefix.append(format_one_labeled_example(item, prompt_template))

    assert len(prefix) > 0
    return "\n".join(prefix)


def load_data(
    labeled_data_filepath: str,
    test_data_filepath: str,
    dataset_name: str,
    n_labeled: int,
    n_test_samples: int,
    inference_mode: InferenceMode,
    label_names: list[str],
):
    """Load the data and generate prompts for inference
    Args:
        labeled_data_filepath: str
            The path to the labeled data
        test_data_filepath: str
            The path to the test data
        dataset_name: str
            The name of the dataset
        n_labeled: int
            The number of labeled examples to use
        n_test_samples: int
            The number of test examples to use
        inference_mode: InferenceMode
            The inference mode to use
        label_names: list[str]
            The names of the labels
    Returns:
        tuple[list[str], list[str], list[int]]
            The prompts, true labels, and sample ids
    """
    data_config_filepath = f"{DATA_CONFIGS_DIRPATH}/{dataset_name}_config.json"
    demonstrations = load_file(labeled_data_filepath)[:n_labeled]
    if n_test_samples == -1:
        test_data = load_file(test_data_filepath)
    else:
        test_data = load_file(test_data_filepath)[:n_test_samples]

    demonstration_prompt_template = load_demontration_prompt_template(data_config_filepath)
    inference_prompt_template = load_inference_prompt_template(data_config_filepath)
    task_instruction = load_instruction(data_config_filepath)

    true_labels = [item["label"] for item in test_data]
    sample_ids = [item["id"] for item in test_data]

    # Prefix contains the labeled examples
    prefix = format_prefix(demonstrations, task_instruction, demonstration_prompt_template)

    test_examples = generate_inference_examples(
        test_data, inference_mode, inference_prompt_template, label_names
    )

    if len(prefix) > 0:  # few-shot case
        prompts = [prefix + "\n" + example for example in test_examples]
    else:  # zero-shot case
        prompts = test_examples

    return prompts, true_labels, sample_ids


def get_label_tokens(model_name, model_path, dataset_name, label_names) -> dict[str, int]:
    """Get the token ids for the labels in the dataset from a pre-existing file. Generate them and store them if they don't exist.
    Args:
        model_name: str
            The name of the model
        model_path: str
            The path to the model. Used only if we need to load the tokenizer model.
        dataset_name: str
            The name of the dataset
        label_names: list[str]
            The names of the labels
    Returns:
        dict[str, int]
            A map from a label to its token id
    """
    label_tokens_filepath = f"{LABEL_TOKENS_DIRPATH}/{model_name}_{dataset_name}_label_tokens.json"
    if os.path.exists(label_tokens_filepath):
        with open(label_tokens_filepath, "r") as f:
            label_tokens = json.load(f)
            print(f"Loaded label tokens: {label_tokens}")
    else:
        print(f"Label tokens file not found, generating for model: {model_name}")
        hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Encode each label and skip the first token because it's the prefix space
        label_tokens = {label: hf_tokenizer.encode(label)[1:] for label in label_names}

        print(f"Generated file tokens: {label_tokens}")

        # Store them in a dictionary for easy access
        with open(label_tokens_filepath, "w") as f:
            json.dump(label_tokens, f)
        print(f"Stored file tokens to: {label_tokens_filepath}")

    return label_tokens


def get_label_names(dataset_name: str) -> list[str]:
    """Return the label names for the dataset from the config.json file.
    Args:
        dataset_name: str
            str: The name of the dataset
    Returns:
        list[str]
            The names of the labels
    """
    with open(f"{DATA_CONFIGS_DIRPATH}/{dataset_name}_config.json", "r") as f:
        config = json.load(f)
    return config["options"]


def load_instruction(data_config_filepath: str) -> str:
    """Load the task instruction from the data config file
    Args:
        data_config_filepath: str
            The path to the data config JSON file
    Returns:
        str
            The task instruction that comes before the examples in the prompt
    """
    with open(data_config_filepath, "r") as f:
        config = json.load(f)
    return config["instruction"]


def load_file(filename: str) -> list[dict]:
    """Load a file into a list of dictionaries
    Args:
        filename: str
            The path to the file
    Returns:
        list[dict]
            The list of data points as dictionaries
    """
    data = []
    if filename.endswith(".jsonl"):
        with open(filename) as fin:
            for l in fin:
                data.append(json.loads(l))
    elif filename.endswith(".csv"):
        data = pd.read_csv(filename, encoding="utf-8").to_dict("records")

    # Ensure we are interleaving 0/1 examples, one after the other, for ease of later processsing
    data_0 = [d for d in data if d["label"] == 0]
    data_1 = [d for d in data if d["label"] == 1]
    data = []
    for d0, d1 in zip(data_0, data_1):
        data.append(d0)
        data.append(d1)

    return data
