import os
import time

from vllm import LLM

from data_utils import (
    get_label_names,
    load_data,
    get_label_tokens,
)
from llm_inference_utils import (
    get_model_config,
    get_sampling_params,
    do_inference,
)
from results_processing import (
    generate_experiment_name,
    save_experiment_results_to_csv,
    store_confusion_matrices,
    evaluate,
)
from parse_args import get_exp_args

DATASETS_DIRPATH = "data/"
UNPROCESSED_PREDS_DIRPATH = "preds/"


def main():
    exp_args = get_exp_args()

    print("Loading model...")
    model_config = get_model_config(exp_args.model_name)
    start = time.time()
    sampling_params = get_sampling_params(exp_args.inference_mode)
    llm = LLM(**model_config)
    model_loading_time = time.time() - start

    print("=" * 80)
    print(f"Running experiments for {exp_args.model_name} on {exp_args.dataset_name}...")

    print("Loading label tokens...")
    train_dataset_path = exp_args.train_filepath
    test_data_dirpath = exp_args.test_filepath
    label_names = get_label_names(exp_args.dataset_name)
    label_names.sort()
    label_tokens = get_label_tokens(
        exp_args.model_name, model_config["model"], exp_args.dataset_name, label_names
    )
    # Ensure label tokens are also sorted by label names
    label_tokens = dict(sorted(label_tokens.items(), key=lambda x: x[0]))

    for n_shots in exp_args.shots_per_class:
        n_labeled = len(label_names) * n_shots

        exp_name = generate_experiment_name(
            inference_mode=exp_args.inference_mode,
            n_test_samples=exp_args.n_test_samples,
            exp_name_details=exp_args.exp_notes,
            dataset_name=exp_args.dataset_name,
            model_name=exp_args.model_name,
            shots_per_class=n_shots,
        )

        preds_output_filepath = f"{UNPROCESSED_PREDS_DIRPATH}/{exp_name}_preds.jsonl"

        # Check that the demonstrations file exists - sometimes we only have a single demonstrations file for a task
        if not os.path.exists(train_dataset_path):
            print(f"File {train_dataset_path} does not exist. Skipping...")
            continue

        print("Loading data...")
        start = time.time()
        prompts, true_labels, sample_ids = load_data(
            labeled_data_filepath=train_dataset_path,
            test_data_filepath=test_data_dirpath,
            dataset_name=exp_args.dataset_name,
            n_labeled=n_labeled,
            n_test_samples=exp_args.n_test_samples,
            inference_mode=exp_args.inference_mode,
            label_names=label_names,
        )
        data_loading_time = time.time() - start

        start = time.time()
        predictions, true_labels, labels_notin_top100, ood_predictions = do_inference(
            inference_mode=exp_args.inference_mode,
            llm=llm,
            sampling_params=sampling_params,
            prompts=prompts,
            preds_output_filepath=preds_output_filepath,
            true_labels=true_labels,
            sample_ids=sample_ids,
            label_tokens=label_tokens,
        )
        generation_time = time.time() - start

        uncalibrated_acc, micro_f1, weighted_f1 = evaluate(true_labels, predictions)

        # Store confusion matrices
        store_confusion_matrices(
            exp_name,
            true_labels,
            predictions,
            label_names,
        )

        # Store results in a csv
        save_experiment_results_to_csv(
            data_loading_time=data_loading_time,
            model_loading_time=model_loading_time,
            exp_name=exp_name,
            generation_time=generation_time,
            uncalibrated_acc=uncalibrated_acc,
            micro_f1=micro_f1,
            weighted_f1=weighted_f1,
            ood_predictions=ood_predictions,
            labels_notin_top100=labels_notin_top100,
            inference_mode=exp_args.inference_mode.value,
            n_test_samples=exp_args.n_test_samples,
            dataset_name=exp_args.dataset_name,
            model_name=exp_args.model_name,
            shots_per_class=n_shots,
            seed=exp_args.seed,
        )
        del llm


if __name__ == "__main__":
    main()
