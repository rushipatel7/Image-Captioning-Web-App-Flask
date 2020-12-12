from flask import Flask, render_template, request
from keras.preprocessing.sequence import pad_sequences
import cv2
import numpy as np
# from keras.utils import to_categorical
# from keras.utils import plot_model
# from keras.models import Model, Sequential, load_model
# from keras.layers import Input
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Embedding
# from keras.layers import Dropout
# from keras.layers.merge import add
# from keras.callbacks import ModelCheckpoint

# from keras.layers import Dense, Flatten, Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, \
    # Activation, RepeatVector, Concatenate
# from  keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, \
    Activation, RepeatVector, Concatenate
from keras.models import Sequential, Model
from tqdm import tqdm

# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# if you has not resnet model then write below line
from tensorflow.keras.applications import ResNet50

resnet = ResNet50(include_top=False, weights='imagenet',
                  input_shape=(224, 224, 3), pooling='avg')

# if you have resnet model then --> directly
# resnet = load_model('resnet.h5')

print("=" * 50)
print("Resnet loaded")

vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()
inv_vocab = {v: k for k, v in vocab.items()}

embedding_size = 128
max_len = 40
vocab_size = len(vocab)

image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

# image_model.summary()

language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

# language_model.summary()

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation(activation='softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs=out)

# model.load_weights("../input/model_weights.h5")
# model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
# model.summary()

model.load_weights('mine_model_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

print("=" * 50)
print("model loaded")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1  # means Every time refresh the page and transfer the file every time


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, vocab, inv_vocab
    file = request.files['file1']

    file.save('./static/file.jpg')

    img = cv2.imread('./static/file.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224,))
    img = np.reshape(img, (1, 224, 224, 3))

    incept = resnet.predict(img).reshape(1, 2048)

    text_in = ['startofseq']
    final = ''

    print("=" * 50)
    print("Grtting Caption")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            # print(vocab[i])
            encoded.append(vocab[i])

        # padded = pad_sequences([encoded])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word
        text_in.append(sampled_word)

    # return render_template('predict.html')
    return render_template('predict.html', final= final)


if __name__ == "__main__":
    app.run(debug=True)
