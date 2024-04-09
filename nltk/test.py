import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from time import time
from flask import Flask
from flask import jsonify

tfidf = TfidfVectorizer(max_features=2000)
df = pd.read_csv("anime.csv")

def set_correct_values():
 df["genre"] = df["genre"].str.replace(","," ")
 df["name"] = df["name"].str.lower().replace(".","")
 df["output"] = df[["genre"]].apply(lambda row: " ".join(row.values.astype(str)),axis=1)
 
def train_model(anime_name: str):
 X = tfidf.fit_transform(df["output"])

 print(anime_name in df["name"])

 if anime_name in df["name"].values:
  animes = pd.Series(df.index,index=df["name"])
  index = animes[anime_name.lower()]
  question = X[index]
  question.toarray()
  simlity = cosine_similarity(question,X)
  simlity = simlity.flatten()
  recomendation = (-simlity).argsort()[1:20]
  df["name"].iloc[recomendation]
  print(df["name"].iloc[recomendation])
  return df["name"].iloc[recomendation]
 return None

def save_model(name_model):
 joblib.dump(tfidf,"{}.pkl".format(name_model))
set_correct_values()

app = Flask(__name__)

@app.route("/api/bot/<anime>",methods=["GET"])
def get_anime_recomendations(anime):
 anime_recomendations = train_model(anime)
 response ={}
 print(anime_recomendations)
 if anime_recomendations is not None:
  response = {"message":"success","results":anime_recomendations.to_list()}
 else:
  response ={"message":"Not found", "results":[]} 
 return jsonify(response) 

app.run(debug=True)