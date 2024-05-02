import numpy as np
import json
from vllm import SamplingParams, LLM
from enum import Enum
from tqdm import tqdm
from results_processing import write_ood_and_missing_labels_to_file

MODELS_DOWNLOAD_DIRPATH = "./models"


class InferenceMode(Enum):
    GREEDY = "greedy"
    TOP100 = "top100"


def do_inference(
    inference_mode: InferenceMode,
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    preds_output_filepath: str,
    true_labels: list[str],
    sample_ids: list[str],
    label_tokens: dict[str, list[int]],
) -> tuple[list[str], list[int], list[tuple[int, str]], list[tuple[int, str]]]:
    """Perform inference on the model.
    Args:
        inference_mode: InferenceMode
            The mode of inference
        llm: LLM
            The language model
        sampling_params: SamplingParams
            The sampling parameters
        prompts: list[str]
            The prompts for the model
        preds_output_filepath: str
            The path to the file where the predictions will be written
        true_labels: list[str]
            The true labels
        sample_ids: list[str]
            The sample ids
        label_tokens: dict[str, list[int]]
            A map from a label to its token ids
    Returns:
        tuple[list[str], list[str], list[tuple[int, str]], list[tuple[int, str]]]
            The predictions, true labels, labels not in the top 100 tokens, and OOD predictions
    """
    label2idx = {label: idx for idx, label in enumerate(list(label_tokens.keys()))}

    print("Generating outputs...")
    outputs = llm.generate(prompts, sampling_params)

    predictions = []
    ood_predictions = []
    labels_notin_top100 = []

    if inference_mode == InferenceMode.GREEDY:
        print("Greedy inference")

        for i, output in enumerate(tqdm(outputs)):
            generated_text = output.outputs[0].text.strip()
            if generated_text not in label2idx:
                ood_predictions.append((i, generated_text))

            # Take into account case where instead of the model generates "\n" instead of "Input:", as if it were to write a new example
            if "Input:" in generated_text:
                pred = generated_text.split("Input:")[0]
            else:
                pred = generated_text

            pred_obj = {
                "generated_text": generated_text,
                "pred": pred,
                "true_label": true_labels[i],
                "id": sample_ids[i],
            }

            with open(preds_output_filepath, "a") as f:
                f.write(json.dumps(pred_obj) + "\n")

            predictions.append(pred)

        print(f"Number of OOD predictions: {len(ood_predictions)}")

        write_ood_and_missing_labels_to_file(
            preds_output_filepath, ood_predictions, labels_notin_top100
        )

    elif inference_mode == InferenceMode.TOP100:
        print("Top100 inference")

        for i, output in enumerate(tqdm(outputs)):
            generated_text = output.outputs[0].text
            logprobs = get_logprobs_from_output(output.outputs[0], label_tokens)
            label_probs, missing_labels = get_label_probs(logprobs, label2idx)
            if missing_labels:
                if len(missing_labels) == len(label2idx):
                    ood_predictions.append((i, generated_text))
                for missing_label in missing_labels:
                    labels_notin_top100.append((i, missing_label))

            idx2label = {v: k for k, v in label2idx.items()}
            pred = idx2label[np.argmax(label_probs)].replace("_", "")
            pred_obj = {
                "generated_text": generated_text,
                "pred": pred,
                "true_label": true_labels[i],
                "id": sample_ids[i],
                "logprobs": logprobs,
            }

            with open(preds_output_filepath, "a") as f:
                f.write(json.dumps(pred_obj) + "\n")

            predictions.append(pred)

        print(
            f"Number of OOD predictions: {len(ood_predictions)}, number of labels not in top 100 tokens: {len(labels_notin_top100)}"
        )

        write_ood_and_missing_labels_to_file(
            preds_output_filepath, ood_predictions, labels_notin_top100
        )

    else:
        raise ValueError(f"Invalid inference mode: {inference_mode}")

    return predictions, true_labels, labels_notin_top100, ood_predictions


def get_sampling_params(inference_mode: InferenceMode):
    """Get the sampling parameters for the model.
    Args:
        inference_mode: InferenceMode
            The mode of inference
    Returns:
        SamplingParams
            The sampling parameters
    """
    if inference_mode == InferenceMode.GREEDY:
        params = SamplingParams(temperature=0.0, stop="\n")
    elif inference_mode == InferenceMode.TOP100:
        params = SamplingParams(temperature=0.0, logprobs=100, max_tokens=1)
    else:
        raise ValueError(f"Invalid inference mode: {inference_mode}")

    return params


def get_logprobs_from_output(output, label_tokens):
    """Get the logprobs and the probs of the labels from the output of the LLM model.
    Args:
        output: CompletionOutput
            The output of the LLM model for a single example
        label_tokens: dict[str, int]
            A map from a label to its token id
    Returns:
        dict[str, tuple]
            A map from a label to a tuple of (logprob, prob, index in top 100 tokens)
    """
    log_probs = output.logprobs
    # Get the logprobs of the labels
    label_logprobs = {}
    for label, label_tokens in label_tokens.items():
        # We're only interested in the first token of the label, since this should only be called for single token labels
        label_token = label_tokens[0]
        # We're iterating through the dictionary because it is in descending order of logprob, so we can also get the index
        for i, token in enumerate(log_probs[0].keys()):
            if label_token == token:
                label_logprobs[label] = (
                    log_probs[0][label_token],  # .logprob,
                    np.exp(log_probs[0][label_token]),  # .logprob),
                    i,
                )  # logprob, index in top 100 tokens

    return dict(sorted(label_logprobs.items(), key=lambda kv: kv[1][-1]))


def get_label_logprobs_from_prompt(label_token_values, prompt_logprobs) -> list[tuple]:
    """Get the logprobs of the labels from the prompt logprobs.
    Args:
        label_token_values: list[int]
            The token values of a given label
        prompt_logprobs: list[dict[int, float]]
            A list of dictionaries, where each dictionary contains a map from a token to its logprob
    Returns:
        list[tuple]
            A list of tuples, where each tuple contains a label token and its logprob
    """
    label_logprobs = []  # [(token: logprob)]

    # Get the last N elements of the prompt logprobs list, where N is the number of tokens in the word we're looking for
    n_last_logprobs = prompt_logprobs[-len(label_token_values) :]
    # print(f"Prompt logprobs: {n_last_logprobs}")
    for i, tokendict in enumerate(n_last_logprobs):

        # Sanity check, technically we should never hit this
        if label_token_values[i] not in tokendict:
            print(f"{i=}")
            print(f"Prompt logprobs: {tokendict}")
            print(f"Label tokens: {label_token_values}")
            raise ValueError(f"Token {label_token_values[i]} not found in the prompt logprobs")

        label_logprobs.append((label_token_values[i], tokendict[label_token_values[i]]))

    return label_logprobs


def get_label_probs(logprobs, label2idx):
    """Get the non-normalized predicted probabilities for each label.
    Args:
        logprobs: dict[str, tuple]
            A map from a label to a tuple of (logprob, prob, index in top 100 tokens)
        label2idx: dict[str, int]
            A map from a label to its index in the list of labels.
    Returns:
        list[list[float]], list[tuple]
            A list of lists, where each list contains the non-normalized predicted probabilities for each label.
            Also returns a list of tuples containing the indices of the test samples that had missing labels, plus the missing label.
    """
    missing_labels = []

    # Get the probs only and ensure that all labels are within the top 100 tokens in the prediction
    pred_probs = [0] * len(label2idx)
    for label in label2idx:
        if label in logprobs:
            pred_probs[label2idx[label]] = logprobs[label][1]  # non-normalized
        else:
            # print(f"Label not found: {label}")
            missing_labels.append(label)

    return pred_probs, missing_labels


def get_model_config(model_name: str):
    """Get the configuration for the model.
    Args:
        model_name: str
            The name of the model
    Returns:
        dict
            The configuration for the model
    """
    model_config = {}
    if model_name == "mixtral-8x7b":
        # gpu_memory_utilization=0.6, tensor_parallel_size=4
        model_config = {
            "model": "mistralai/Mixtral-8x7B-v0.1",
            "gpu_memory_utilization": 0.6,
            "tensor_parallel_size": 4,
        }
    elif model_name == "gemma-7b":
        # gpu_memory_utilization=0.45, max_model_len=4096
        model_config = {
            "model": "google/gemma-7b",
            "gpu_memory_utilization": 0.45,
            "max_model_len": 4096,
        }
    elif model_name == "mistral-7b":
        # gpu_memory_utilization=0.7
        model_config = {
            "model": "mistralai/Mistral-7B-v0.1",
            "gpu_memory_utilization": 0.7,
        }
    elif model_name == "LLaMA2-13B":
        # tensor_parallel_size=2
        model_config = {
            "model": "meta-llama/Llama-2-13b-hf",
            "tensor_parallel_size": 2,
        }
    elif model_name == "LLaMA2-70B":
        # gpu_memory_utilization=0.85, tensor_parallel_size=4
        model_config = {
            "model": "meta-llama/Llama-2-70b-hf",
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 4,
        }
    elif model_name == "LLaMA2-7B":
        model_config = {
            "model": "meta-llama/Llama-2-7b-hf",
        }
    else:
        raise ValueError(f"Model name not recognized: {model_name}")

    model_config["download_dir"] = MODELS_DOWNLOAD_DIRPATH
    return model_config
