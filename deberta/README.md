# CS567_ML_Project
LLM detect AI generated text using DeBERTa model

## Installation and Setup
Follow these steps to install the required packages for your environment:
```bash
cd src
pip install -r requirements.txt
```
## Training and Running DeBERTa

Open 'llm-detect-train-val-deberta.ipynb', the first cell contains a sample of different datasets. 
Whichever one you choose, make sure to update the train_test_split function:
```bash
# Split data into train and validation sets
train_data, val_data = train_test_split([insert_data_here], test_size=0.3, random_state=42)
