import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from llm_inference_utils import InferenceMode
import numpy as np
from sklearn.metrics import f1_score


RESULTS_CSV_FILEPATH = "./results.csv"
CONFUSION_MATRICES_DIRPATH = "./confusion_matrices"


def evaluate(true_labels, predictions):
    print("Evaluating...")
    uncalibrated_acc = np.mean(
        [1 if pred == true_label else 0 for pred, true_label in zip(predictions, true_labels)]
    )
    micro_f1 = f1_score(true_labels, predictions, average="micro")
    weighted_f1 = f1_score(true_labels, predictions, average="weighted")

    print(f"Uncalibrated accuracy: {uncalibrated_acc:.4f}")
    print(f"Uncalibrated micro F1 score: {micro_f1:.4f}")
    print(f"Uncalibrated weighted F1 score: {weighted_f1:.4f}")
    return uncalibrated_acc, micro_f1, weighted_f1


def store_confusion_matrices(exp_name, true_labels, predictions, label_names):
    cm_normalized = confusion_matrix(true_labels, predictions, labels=label_names, normalize="true")

    disp_normalized = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=label_names
    )
    disp_normalized.plot()
    plt.title("Normalized confusion matrix")
    plt.xticks(rotation=45)
    plt.savefig(f"{CONFUSION_MATRICES_DIRPATH}/{exp_name}_cm_.png")


def generate_experiment_name(
    inference_mode: InferenceMode,
    n_test_samples: int,
    exp_name_details: str,
    dataset_name: str,
    model_name: str,
    shots_per_class: int,
) -> str:
    new_exp_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name.lower()}_{dataset_name.lower().replace('_', '-')}_shots-per-class={shots_per_class}-inf-mode={inference_mode.value}"

    if n_test_samples != -1:
        new_exp_name += f"-n-test={n_test_samples}"

    if exp_name_details:
        new_exp_name += f"-details={exp_name_details}"

    return new_exp_name


def save_experiment_results_to_csv(
    data_loading_time: float,
    model_loading_time: float,
    exp_name: str,
    generation_time,
    uncalibrated_acc: float,
    micro_f1: float,
    weighted_f1: float,
    inference_mode: str,
    n_test_samples: int,
    dataset_name: str,
    model_name: str,
    shots_per_class: int,
    seed: int,
    ood_predictions: list[tuple] = None,
    labels_notin_top100: list[tuple] = None,
):
    if os.path.exists(RESULTS_CSV_FILEPATH):
        results_df = pd.read_csv(RESULTS_CSV_FILEPATH)
    else:
        results_df = pd.DataFrame()

    cur_run_df = pd.DataFrame(
        {
            "exp_name": [exp_name],
            "uncalibrated_acc": [round(uncalibrated_acc, 4)],
            "uncalibrated_f1_micro": [round(micro_f1, 4)],
            "uncalibrated_f1_weighted": [round(weighted_f1, 4)],
            "ood_predictions": [len(ood_predictions) if ood_predictions else None],
            "missing_labels": [len(labels_notin_top100) if labels_notin_top100 else None],
            "data_loading_time": [round(data_loading_time, 6)],
            "model_loading_time": [round(model_loading_time, 6)],
            "generation_time": [round(generation_time, 6)],
            "inference_mode": [inference_mode],
            "n_test_samples": [n_test_samples],
            "dataset_name": [dataset_name.lower()],
            "model_name": [model_name.lower()],
            "shots_per_class": [shots_per_class],
            "seed": [seed],
        }
    )

    # Concatenate results
    results_df = pd.concat([results_df, cur_run_df], ignore_index=True)
    results_df.to_csv(RESULTS_CSV_FILEPATH, index=False)


def write_ood_and_missing_labels_to_file(
    preds_output_filepath, ood_predictions, labels_notin_top100
):
    preds_output_filepath = preds_output_filepath.replace(".jsonl", "-ood-missing-labels.txt")

    # Store ood predictions in the same predictions file
    if ood_predictions:
        with open(preds_output_filepath, "a") as f:
            f.write(f"{'-' * 80}\n")
            f.write("Out-of-distribution predictions:\n")
            for i, pred in ood_predictions:
                f.write(f"i: {i}, ood_pred: {pred}\n")
            f.write(f"{'-' * 80}\n")

    # Store number of labels not in the top 100 tokens in the same predictions file
    if labels_notin_top100:
        with open(preds_output_filepath, "a") as f:
            f.write(f"{'-' * 80}\n")
            f.write("Labels not in top 100 tokens:\n")
            for i, label in labels_notin_top100:
                f.write(f"pred idx: {i}, label: {label}\n")
            f.write(f"{'-' * 80}\n")
