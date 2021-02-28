# A simple chatbot with tensorflow

Inpired on [python-engineer/pytorch-chatbot](https://github.com/python-engineer/pytorch-chatbot)

## Installation
MacOS/Linux

### Create an environment
Using conda:
```
conda create --name chatbot python==3.7
conda activate chatbot
conda install tensorflow nltk numpy
# or:
# conda install --file requirements.txt
```

Using venv:
```
python3.7 -m venv venv
. ./venv/bin/activate
pip install tensorflow nltk numpy
# or:
# pip install -r requirements.txt
```

## Usage

Run
```
python train.py
```

This will train and export the trained model into a `model_trained` directory and dump `disctionary.json` file. Then run

```
python chat.py
```