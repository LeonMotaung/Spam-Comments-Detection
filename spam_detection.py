import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Load the dataset
data = pd.read_csv("Youtube01-Psy.csv")
print(data.sample(5))

data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam Comment"})

x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
nb = BernoulliNB()
model = VotingClassifier(estimators=[('rf', rf), ('nb', nb)], voting='hard')

model.fit(xtrain, ytrain)
print("Ensemble Model Accuracy:", model.score(xtest, ytest))

sample = "Check this out: https://amanxai.com/" 
data_sample = cv.transform([sample]).toarray()
print(model.predict(data_sample))

sample = "Lack of information!" 
data_sample = cv.transform([sample]).toarray()
print(model.predict(data_sample)) 
