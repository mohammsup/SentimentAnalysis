import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords

df = pd.read_csv('./GBcomments.csv', nrows=100)
df = df[["likes","comment_text"]]
df.head()

df.comment_text = df.comment_text.apply(lambda x: x.lower())
df.head()

df['comment_text'] = df['comment_text'].str.replace('[^\w\s]','')
df.head()

def num_split(text):
    x = re.sub("\d+", "", text)
    return x
df["comment_text"] = df["comment_text"].apply(num_split)

df

stop = stopwords.words('english')
df['comment_text'] = df['comment_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

df.head()

from textblob import TextBlob

df['comment_text'][:5].apply(lambda x: TextBlob(x).sentiment)

def detect_polarity(text):
    x = TextBlob(text).sentiment.polarity
    if x>0:
        x="positive"
    elif x<0:
        x="negative"
    else:
        x="neutral"
    return x
df['label'] = df.comment_text.apply(detect_polarity)
df.head()

#trainData = df[:80]
#testData = df[81:100]
#trainData.head()

comment_text = df.comment_text.str.cat(sep=' ')
tokens = word_tokenize(comment_text)
vocabulary = set(tokens)
frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]
print(len(vocabulary))

stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]

X_train = df.loc[:80, 'comment_text'].values
y_train = df.loc[:80, 'label'].values
X_test = df.loc[81:100, 'comment_text'].values
y_test = df.loc[81:100, 'label'].values

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_vectors, y_train)

from  sklearn.metrics  import accuracy_score
predicted = clf.predict(test_vectors)
print(accuracy_score(y_test,predicted))