from dataclasses import dataclass
from typing import List, Optional
import argparse


@dataclass
class ExperimentArgs:
    seed: int
    model_name: str
    dataset_name: str
    train_filepath: str
    test_filepath: str
    exp_notes: Optional[str]
    n_test_samples: int
    evaluate_on: str
    inference_mode: str
    shots_per_class: List[int]


def get_exp_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=("LLaMA2-7B", "mistral-7b", "mixtral-8x7b", "gemma-7b", "LLaMA2-13B", "LLaMA2-70B"),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=("daigt", "hewlett", "xsum"),
    )
    parser.add_argument("--train_filepath", type=str, required=True)
    parser.add_argument("--test_filepath", type=str, required=True)
    parser.add_argument("--exp_notes", type=str, default=None)
    parser.add_argument(
        "--n_test_samples", type=int, required=True, help="Write -1 for testing on all samples"
    )
    parser.add_argument("--evaluate_on", type=str, default="test", choices=("dev", "test"))

    parser.add_argument(
        "--inference_mode",
        type=str,
        default="top100",
        choices=("greedy", "top100"),
        help="Inference mode to use for generating predictions",
    )

    parser.add_argument(
        "--shots_per_class",
        type=int,
        nargs="*",
        required=True,
        help="Number of labeled data points (shots) per each class to use as in-context learning examples",
    )

    args = parser.parse_args()

    experiment_args = ExperimentArgs(**args)

    print("Arguments:")
    for field, value in vars(experiment_args).items():
        print(f"- {field}: {value}")

    return experiment_args
