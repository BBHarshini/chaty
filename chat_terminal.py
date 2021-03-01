import random
import json
import numpy as np
from nltk_utils import bag_of_words, tokenize
import tensorflow as tf

with open('intents.json', 'r') as f:
    intents = json.load(f)
with open('dictionary.json', 'r') as f:
    data = json.load(f)

input_size = data['input_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']

model = tf.keras.models.load_model('model_trained')

bot_name = "Valentina"
print("Vamos conversar! Digite 'sair' para encerrar")
while True:
    sentence = input("Você: ")
    if sentence == 'sair':
        print(f'Encerrando... Até logo.')
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])

    output = model(X)
    tag = tags[np.argmax(output)]
    # print(tag)

    softmax = tf.keras.layers.Softmax()
    probs = softmax(output).numpy()
    # print(probs)
    prob = probs[0][np.argmax(output)]
    if prob > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Desculpe, não sei nada sobre isso...")
