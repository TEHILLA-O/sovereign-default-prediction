!python --version

!pip install tensorflow

import tensorflow as tf
print(tf.__version__)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from google.colab import files

uploaded = files.upload()

np.random.seed(42)
tf.random.set_seed(42)



df = pd.read_csv('sovereign_default_dataset_1980_2022.csv - sovereign_default_dataset_1980_2022.csv')


df = df.sort_values(['Country', 'Year'])
features = [
    'Debt_GDP', 'ExtDebt_GDP', 'DebtServ_XGDP', 'RealGDP_growth', 'GDP_per_capita_USD',
    'Inflation', 'fiscal_balance', 'primary_balance', 'CurrentAccount_GDP', 'Reserves_months',
    'ExchangeRate_change', 'Trade_openness', 'US_FedFundsRate', 'World_GDP_growth',
    'Oil_price_index', 'VIX', 'ICRG_political', 'ElectionYear'
]

sequence_length = 5
X_sequences, y_labels = [], []

for country in df['Country'].unique():
    country_df = df[df['Country'] == country]
    for i in range(len(country_df) - sequence_length):
        seq_X = country_df.iloc[i:i+sequence_length][features].values
        target_y = country_df.iloc[i+sequence_length]['Default']
        X_sequences.append(seq_X)
        y_labels.append(target_y)

X_sequences = np.array(X_sequences)
y_labels = np.array(y_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

model = Sequential([
    LSTM(64, input_shape=(sequence_length, len(features)), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

from tensorflow.keras.layers import Input

input_layer = Input(shape=(sequence_length, len(features)))
x = LSTM(64, return_sequences=False, name='lstm_layer')(input_layer)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer('lstm_layer').output)
X_train_embed = extractor.predict(X_train)
X_test_embed = extractor.predict(X_test)

X_train_embed_flat = X_train_embed.reshape((X_train_embed.shape[0], -1))
X_test_embed_flat = X_test_embed.reshape((X_test_embed.shape[0], -1))

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
)

xgb_model.fit(X_train_embed_flat, y_train)

y_pred = xgb_model.predict(X_test_embed_flat)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Hybrid model Accuracy: {accuracy:.3f}")
print(f"Hybrid model F1 Score: {f1:.3f}")
