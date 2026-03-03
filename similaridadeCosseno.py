import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

df = pd.read_csv("all_games.csv")

example_name = df["name"].tolist()
example_descrition = df["summary"].tolist()
example_platform = df["platform"].tolist()
example_score = df["meta_score"].tolist()
example_review = df["user_review"].tolist()

stop_words = set(stopwords.words("english"))

vectorizer = CountVectorizer()
vectors_count = vectorizer.fit_transform(df["summary"].values.astype("U"))

user = input("Digite qual o jogo que você procura similares:\n")
vector = vectorizer.transform([user])

similarity = cosine_similarity(vector, vectors_count)
top_similares = similarity[0].argsort()[::-1][:5]


print("\nOs cincos games mais similares são")
for i, indice in enumerate(top_similares):
    print(f"\nGame:{example_name[indice]}")
    print(f"\nDescrição:{example_descrition[indice]}")
    print(f"\nPlataforma:{example_platform[indice]}")
    print(f"\nPontuação:{example_score[indice]}")
    print(f"\nAvaliação:{example_review[indice]}")
    angulo_cos = np.arccos(similarity[0][indice])
    angulo_graus = np.degrees(angulo_cos)
    print(f"\nAngulo do Cosseno(em graus):{angulo_graus:.2f} graus\n")
