# A simple chatbot with tensorflow

Inpired by [python-engineer/pytorch-chatbot](https://github.com/python-engineer/pytorch-chatbot)

## Installation
MacOS/Linux

### Create an environment
Using conda:
```
conda create --name chatbot python==3.7
conda activate chatbot
conda install --file requirements.txt
```

Using venv:
```
python3.7 -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
```

## Usage

Training the model
```
python train.py
```

This will train and export the trained model into a `model_trained` directory and dump a `dictionary.json` file.

Running the API (development mode)
```
uvicorn api:app --reload
```
