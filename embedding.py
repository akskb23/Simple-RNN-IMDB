
from tensorflow.keras.preprocessing.text import one_hot

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

# Define the vocabulary size
voc_size=10000

# One Hot Representation
one_hot_repr=[one_hot(words,voc_size) for words in sent]
# print(one_hot_repr)

# Word Embedding Representation

from tensorflow.keras.layers import Embedding 
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

sent_length = 8
embedded_docs=pad_sequences(one_hot_repr,padding='pre',maxlen=sent_length)
# print(embedded_docs)

# Feature Representation 
dimension=10

model=Sequential()
model.add(Embedding(voc_size,dimension,input_length=sent_length))
model.compile('adam','mse')

print(model.summary())

print(model.predict(embedded_docs))
