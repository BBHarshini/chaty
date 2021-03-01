import os
import json
import numpy as np
import tensorflow as tf
from nltk_utils import tokenize, stem, bag_of_words
from model import create_model
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bow = bag_of_words(pattern_sentence, all_words)
    X_train.append(bow)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
batch_size = 4
shuffle_buffer_size = 4
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
epochs = 100
assert input_size == len(all_words)
assert output_size == len(tags)

dictionary = {
    "input_size": input_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

with open('dictionary.json', 'w') as f:
    json.dump(dictionary, f)

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)

model = create_model(input_size=input_size, output_size=output_size)

optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optim, loss=loss)

model.fit(dataset, epochs=epochs)

model.save('./model_trained')
print(f"training complete. model saved.")
