
from nltk.corpus import movie_reviews


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix



#print(movie_reviews.fileids())


def func(list0):
    sents = []
    for blist in list0:
        fsent = ""
        for slist in blist:
            fsent += slist + " "
        sents.append(fsent)
    return sents


sents0 = movie_reviews.sents("neg/cv000_29416.txt")
sents1 = movie_reviews.sents("pos/cv041_21113.txt")

texts0 = func(sents0)
texts1 = func(sents1)

file = open("table.txt", "w")
for i in texts0:
    file.write("neg" + "\t" + i + "\n")
file.close()

print(file)

file = open("table.txt", "a")
for i in texts1:
    file.write("pos" + "\t" + i + "\n")
file.close()

print(file)


df = pd.read_table("table.txt" , sep='\t', header=None, names=['label', 'message'])
df['label'] = df.label.map({'neg': 1, 'pos': 0})
# df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()

counts = count_vect.fit_transform(df['message'])

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=72)

model = MultinomialNB().fit(X_train, y_train)

#print(X_train, X_test, y_train, y_test)

predicted = model.predict(X_test)

print(np.mean(predicted == y_test))

