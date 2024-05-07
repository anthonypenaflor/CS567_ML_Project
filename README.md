# CS567 Spring 2025 Machine Learning Project: Detecting AI Generated Text"

## Setup

```shell
pip install requirements.txt
```

## Generating additional data

### XSUM

Parameters for controlling generation can be modified inside `generate_xsum_data.py`

```shell
cd data_generation
python generate_xsum_data.py
```

### Hewlett

Parameters for controlling generation can be modified inside `generate_hewlett_data.py`

```shell
cd data_generation
python generate_hewlett_data.py
```

## Running the models

### LLM-ICL

The code to run in-context learning classification using Llama3-8b is in the `llm-icl.ipynb` notebook. It can be ran on Colab, but it requires the data to be mounted in your Google Drive.


