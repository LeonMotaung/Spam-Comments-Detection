Spam comments detection means classifying comments as spam or not spam. YouTube is one of the platforms that uses Machine Learning to filter spam comments automatically to save its creators from spam comments. If you want to learn how to detect spam comments with Machine Learning, this article is for you. In this article, I will take you through the task of Spam comments detection with Machine Learning using Python.

Spam Comments Detection
Detecting spam comments is the task of text classification in Machine Learning. Spam comments on social media platforms are the type of comments posted to redirect the user to another social media account, website or any piece of content.

To detect spam comments with Machine Learning, we need labelled data of spam comments. Luckily, I found a dataset on Kaggle about YouTube spam comments which will be helpful for the task of spam comments detection. You can download the dataset from here.

In the section below, you will learn how to detect spam comments with machine learning using the Python programming language.

## Spam Comments Detection using Python
Let’s start this task by importing the necessary Python libraries and the dataset:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

data = pd.read_csv("Youtube01-Psy.csv")
print(data.sample(5))

# We need to map comments class labels 
data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam Comment"})

x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
nb = BernoulliNB()
model = VotingClassifier(estimators=[('rf', rf), ('nb', nb)], voting='hard')

model.fit(xtrain, ytrain)
print("Ensemble Model Accuracy:", model.score(xtest, ytest))
```

Now let’s test the model by giving spam and not spam comments as input:
```python
sample = "Check this out: https://amanxai.com/" 
data_sample = cv.transform([sample]).toarray()
print(model.predict(data_sample))
```
`['Spam Comment']`

```python
sample = "Lack of information!" 
data_sample = cv.transform([sample]).toarray()
print(model.predict(data_sample)) 
```
`['Not Spam']`

So this is how you can train a Machine Learning model for the task of spam detection using Python.
