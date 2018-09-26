
import tensorflow as tf
from tensorflow import keras

import numpy as np
imdb = keras.datasets.imdb

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

len(train_data[0]), len(train_data[1])



# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

len(train_data[0]), len(train_data[1])

print(train_data[0])


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
###########################
###applying the model build for movie review to restaurant reviews
#https://github.com/Hvass-Labs/TensorFlow-Tutorials
##https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/20_Natural_Language_Processing.ipynb
import pandas as pd

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
hotel_rv=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
##train model

###test data set
data_text=hotel_rv.Review
data_y=hotel_rv.Liked
########
x_train_text=data_text[:800]
y_train=data_y[:800]

x_test_text=data_text[801:999]
y_test = data_y[801:999]
print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data_text)

if num_words is None:
    num_words = len(tokenizer.word_index)
    

tokenizer.word_index  
  
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)

x_train_text[1]

np.array(x_train_tokens[1])

x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

np.mean(num_tokens)

np.max(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens

np.sum(num_tokens < max_tokens) / len(num_tokens)

pad = 'pre'



x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)


x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

x_train_pad.shape

x_test_pad.shape

np.array(x_train_tokens[1])


x_train_pad[1]

idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]
    
    # Concatenate all words.
    text = " ".join(words)

    return text


x_train_text[1]


tokens_to_string(x_train_tokens[1])

model = Sequential()

embedding_size = 8

model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))

model.add(GRU(units=16, return_sequences=True))


model.add(GRU(units=8, return_sequences=True))

model.add(GRU(units=4))

model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


model.summary()



model.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=3, batch_size=64)

result = model.evaluate(x_test_pad, y_test)

print("Accuracy: {0:.2%}".format(result[1]))

y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])

cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]


len(incorrect)

idx = incorrect[0]
idx

text = x_test_text[idx]
text

##########
y_test=hotel_rv.Liked
tokenizer.fit_on_texts(data_text)

if num_words is None:
    num_words = len(tokenizer.word_index)
    
tokenizer.word_index    
 
x_test_tokens = tokenizer.texts_to_sequences(data_text) 

num_tokens = [len(tokens) for tokens in  x_test_tokens]
num_tokens = np.array(num_tokens)  
np.mean(num_tokens)
np.max(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens

np.sum(num_tokens < max_tokens) / len(num_tokens)
pad = 'pre'

x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
x_test_pad.shape
y_test
result = model.evaluate(x_test_pad, y_test)
print("Accuracy: {0:.2%}".format(result[1]))
y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])

cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

len(incorrect)
idx = incorrect[0]

idx

text = x_test_text[idx]
text

text5="Food was nice"
text6="Taste was pathetic."
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
texts = [ text5, text6, text7, text8]

tokens = tokenizer.texts_to_sequences(texts)

tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
tokens_pad.shape

model.predict(tokens_pad)

idx