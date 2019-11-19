import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('./GBcomments.csv', nrows=100)
df = df[["likes","comment_text"]]
df.head()

def detect_polarity(text):
    x = TextBlob(text).sentiment.polarity
    return x
df['label'] = df.comment_text.apply(detect_polarity)
df.head()

X, y = (df['comment_text'].values, df['label'].values)

tk = Tokenizer(lower = True,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tk.fit_on_texts(X)
X_seq = tk.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.25, random_state = 1)

batch_size = 32
X_train1 = X_train[batch_size:]
y_train1 = y_train[batch_size:]
X_valid = X_train[:batch_size]
y_valid = y_train[:batch_size]

vocabulary_size = len(tk.word_counts.keys())+1
max_words = 100

embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(200))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train1,y_train1,validation_data=(X_valid,y_valid),batch_size=batch_size,epochs=3)

score = model.evaluate(X_test,y_test,verbose=0)
score