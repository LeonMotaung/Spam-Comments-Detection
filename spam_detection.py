import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Load the dataset
data = pd.read_csv("Youtube01-Psy.csv")
print(data.sample(5))
