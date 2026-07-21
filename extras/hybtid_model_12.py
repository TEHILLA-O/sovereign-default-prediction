from google.colab import files
uploaded = files.upload()

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('sovereign_default_dataset_1980_2022.csv - sovereign_default_dataset_1980_2022.csv')

sequence_length = 5
features = [col for col in df.columns if col not in ['Country', 'Year', 'Default']]
X_seq = []
y_labels = []

countries = df['Country'].unique()
for country in countries:
    country_df = df[df['Country'] == country].sort_values('Year')
    for i in range(len(country_df) - sequence_length):
        seq_X = country_df.iloc[i:i+sequence_length][features].values
        target_y = country_df.iloc[i+sequence_length]['Default']
        X_seq.append(seq_X)
        y_labels.append(target_y)

X_seq = np.array(X_seq)
y_labels = np.array(y_labels)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_labels, stratify=y_labels, test_size=0.2, random_state=42)

num_features = X_train.shape[2]

inputs = Input(shape=(sequence_length, num_features))
x = LSTM(64, return_sequences=False)(inputs)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

embedding_model = Model(inputs=model.input, outputs=model.layers[1].output)
X_train_embed = embedding_model.predict(X_train)
X_test_embed = embedding_model.predict(X_test)

X_train_flat = X_train_embed.reshape(X_train_embed.shape[0], -1)
X_test_flat = X_test_embed.reshape(X_test_embed.shape[0], -1)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_flat, y_train)

y_pred = xgb_model.predict(X_test_flat)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n✅ Hybrid model Accuracy: {acc:.3f}")
print(f"✅ Hybrid model F1 Score: {f1:.3f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_flat)

shap.summary_plot(shap_values, X_test_flat, plot_type="bar")

shap.summary_plot(shap_values, X_test_flat)
