######
from sklearn.feature_extraction.text import TfidfVectorizer
######


vect = TfidfVectorizer(min_df=1)
tfidf = vect.fit_transform(["I like cars",
                             "im an albatroz",
                            "Sweet Child o mine"])
print((tfidf * tfidf.T).A)

print ("Rodou")