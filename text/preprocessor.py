from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import string
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# NUM_WORDS is a variable containing the number of words we want to keep in our vocabulary
NUM_WORDS = 8000

# SEQ_LEN to determine how many words to use from each review
SEQ_LEN = 256

BATCH_SIZE = 16
EPOCHS = 10
EMBEDDING_SIZE = 16
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, 'model/model.h5')
TOKENIZER_PICKLE_DIR = os.path.join(BASE_DIR, 'model/tokenizer.pickle')

'''
create tensorflow model
'''
def create_model():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_SIZE))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():

    train_dataframe = get_data_frame('train')
    test_dataframe = get_data_frame('test')

    # encode data to numeric
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS, oov_token='<UNK>')
    tokenizer.fit_on_texts(train_dataframe['text'])
    print(type(tokenizer))

    # convert text data to numerical indexes
    train_sequence = tokenizer.texts_to_sequences(train_dataframe['text'])
    test_sequence = tokenizer.texts_to_sequences(test_dataframe['text'])

    train_sequence = tf.keras.preprocessing.sequence.pad_sequences(train_sequence, maxlen=SEQ_LEN, padding='post')
    test_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, maxlen=SEQ_LEN, padding='post')

    '''
    EarlyStopping callback, which causes the model to stop training if 
    the validation accuracy starts to decrease, which helps reduce overfitting
    '''
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max')

    callbacks = [es]

    model = create_model()
    model.fit(train_sequence, train_dataframe['sentiment'].values
                    , batch_size=BATCH_SIZE
                    , epochs=EPOCHS
                    , validation_split=0.2
                    , callbacks=callbacks)
    save_model(model, tokenizer)

'''
save tensorflow trained model
'''
def save_model(model, tokenizer):
    model.save(MODEL_DIR)

    with open(TOKENIZER_PICKLE_DIR, 'wb') as tp:
        pickle.dump(tokenizer, tp, protocol=pickle.HIGHEST_PROTOCOL)
    
'''
load tensorflow trained model from disk
'''
def load_model():
    with open(TOKENIZER_PICKLE_DIR, 'rb') as tp:
        tokenizer_data = pickle.load(tp)
    return (tf.keras.models.load_model(MODEL_DIR), tokenizer_data)

'''
get data frame from text file
asumming our folder looks like this
data/aclImdb:
    test:
        - neg
        - pos
    train:
        - neg
        - pos
'''
def get_data_frame(path_folder):
    df = pd.DataFrame(columns=['text', 'sentiment'])

    base_folder_data = os.path.join(BASE_DIR, 'data/aclImdb')
    folder_data = os.path.join(base_folder_data, path_folder)

    text = []
    sentiment = []
    for d in ['neg', 'pos']:
        folder = os.path.join(folder_data, d)
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        for f in files:
            with open(os.path.join(folder, f), 'r') as text_file:
                text.append(text_file.read().replace('\n', ' ').replace('\r', ' '))
                sentiment.append(1 if d == 'pos' else 0)
    
    df['text'] = text
    df['sentiment'] = sentiment

    # This line shuffles the data so you don't end up with contiguous
    # blocks of positive and negative reviews
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def test_predict(model, tokenizer, reviews = []):
    sequence = tokenizer.texts_to_sequences(reviews)
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=SEQ_LEN, padding='post')
    results = model.predict(sequence)

    print(results)

    prediction_df = pd.DataFrame(columns=['Reviews', 'Sentiment'])
    prediction_df['Reviews'] = reviews
    prediction_df['Sentiment'] = results

    prediction_df['Sentiment'] = prediction_df['Sentiment'].apply(lambda x: 'Positive' if x >= 0.499 else 'Negative')
    return prediction_df


def main():
    # train_model()

    (model, tokenizer_data) = load_model()
    # model.evaluate(test_sequence, test_dataframe['sentiment'].values)

    my_reviews=['this movie was awesome',
           'this movie was the worst movie ive ever seen',
           'i hated everything about this movie',
           'this is my favorite movie of the year',
           'i dislike this movie, so bad, worst and very very bad',
           'i love and miss this movie so bad'
           ]

    results = test_predict(model, tokenizer_data, my_reviews)
    print(results)

if __name__ == '__main__':
    main()