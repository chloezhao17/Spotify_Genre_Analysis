#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(os.path.join("..", "pull-data", "songs_with_features.csv"))
df.head()


df.columns


df=df.loc[(df['genre']=="blues") | (df['genre']=="soul")| (df['genre']=="jazz")]
df


y = df["genre"]

X = df[["danceability","acousticness","energy","instrumentalness", "loudness","mode","speechiness","tempo"]]
X.head()


# Step 1: Label-encode data set
label_encoder = LabelEncoder()
label_encoder.fit(y)
encoded_y = label_encoder.transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, random_state=42)


train_scores = []
test_scores = []
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
    
    
plt.plot(range(1, 20, 2), train_scores, marker='o')
plt.plot(range(1, 20, 2), test_scores, marker="x")
plt.xlabel("k neighbors")
plt.ylabel("Testing accuracy Score")
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model=Sequential()
number_inputs = 8
number_hidden_nodes = 4
model.add(Dense(units=number_hidden_nodes,
                activation='relu', input_dim=number_inputs))
model.add(Dense(units=number_hidden_nodes,
                activation='relu'))


number_classes = 2
model.add(Dense(units=number_classes, activation='softmax'))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


from tensorflow.keras.utils import to_categorical
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)
y_train_categorical


model.fit(
    X_train,
    y_train_categorical,
    epochs=400,
    shuffle=True
)


model_loss, model_accuracy = model.evaluate(
    X_test, y_test_categorical, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


model_loss, model_accuracy = model.evaluate(
    X_test, y_test_categorical, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")







