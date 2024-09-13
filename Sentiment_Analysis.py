#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')  # Download necessary NLTK data
nltk.download('stopwords')


# In[ ]:


text = "I love this movie!"
tokens = word_tokenize(text)
print(tokens)


# In[ ]:


lowercase_tokens = [token.lower() for token in tokens]
print(lowercase_tokens)


# In[ ]:


stopwords = set(stopwords.words('english'))
filtered_tokens = [token for token in lowercase_tokens if token not in stopwords]
print(filtered_tokens)


# In[ ]:


import re

cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in filtered_tokens]
print(cleaned_tokens)


# In[ ]:


stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
print(stemmed_tokens)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[ ]:


corpus = ["I love this movie!", "This movie is great.", "I don't like this movie."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())


# In[ ]:


corpus = ["I love this movie!", "This movie is great.", "I don't like this movie."]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset from CSV
data = pd.read_csv('data.csv')
X = data['Sentence']
y = data['Sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train the SVM classifier
clf = svm.SVC()
clf.fit(X_train_features, y_train)

# Make predictions
y_pred = clf.predict(X_test_features)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

# Split the dataset into features (X) and labels (y)
sentences = data['Sentence'].values
labels = data['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

# Tokenize the sentences
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)


# In[ ]:


# Pad the sequences
padded_sequences = pad_sequences(sequences, padding='post')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define the neural network architecture
vocab_size = len(tokenizer.word_index) + 1  # Added +1 because of reserved 0 index for padding
embedding_dim = 100  # You can choose any size for the embedding_dim
max_length = len(max(sequences, key=len))

model = keras.Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Load the dataset from CSV
data = pd.read_csv('data.csv')

# Split the dataset into features (X) and labels (y)
X = data['Sentence']
y = data['Sentiment']

# Convert labels to numerical values
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train the model
model = SVC()
model.fit(X_train_features, y_train)

# Make predictions
y_pred = model.predict(X_test_features)


# In[ ]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_matrix)

# Hyperparameter tuning using GridSearchCV
parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(model, parameters, cv=5)
grid_search.fit(X_train_features, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)


# In[ ]:




