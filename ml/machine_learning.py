#!/usr/bin/env python
# coding: utf-8


def run_model(features):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import os
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.utils import to_categorical
    df = pd.read_csv(os.path.join(
        "..", "pull-data", "songs_with_features.csv"))
    df.head()

    # Select Data
    filtered_df = df.loc[df["song_id"] == -1]
    for gen in features:
        filtered_df = filtered_df.append(df.loc[df['genre'] == gen])
    # df = df.loc[(df['genre'] == "blues") | (
    #     df['genre'] == "soul") | (df['genre'] == "jazz")]
    y = filtered_df["genre"]
    X = filtered_df[["danceability", "acousticness", "energy",
                     "instrumentalness", "loudness", "mode", "speechiness", "tempo"]]

    # Create a Model
    model = Sequential()
    number_inputs = 8
    number_hidden_nodes = 4
    model.add(Dense(units=number_hidden_nodes,
                    activation='relu', input_dim=number_inputs))
    model.add(Dense(units=number_hidden_nodes,
                    activation='relu'))
    number_classes = len(features)
    model.add(Dense(units=number_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    encoded_y = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded_y, random_state=42)
    train_scores = []
    test_scores = []
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)
    model.fit(
        X_train,
        y_train_categorical,
        epochs=400,
        shuffle=True,
        verbose=0
    )

    # get model loss and accuracy
    model_loss, model_accuracy = model.evaluate(
        X_test, y_test_categorical, verbose=2)
    # print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
    return(model_loss, model_accuracy)
