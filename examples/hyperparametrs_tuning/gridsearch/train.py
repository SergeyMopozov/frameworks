import config

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import pandas as pd


print("[INFO] loading data...")
dataset = pd.read_csv(config.TRAIN)
X = dataset[dataset.columns[1:-1]]
y = dataset[config.TARGET]
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, random_state=0, test_size=0.25)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)


print("[INFO] training our model")
model = LGBMClassifier(random_state=0)
model.fit(X_train, y_train)
# evaluate our model using logloss-score
print("[INFO] evaluating...")
loss = log_loss(pd.get_dummies(y_train).values, model.predict_proba(X_train))
print(f"Train Log Loss: {loss:.5f}")
loss = log_loss(pd.get_dummies(y_valid).values, model.predict_proba(X_valid))
print(f"Valid Log Loss: {loss:.5f}")


print("[INFO] make submission")
X_test = pd.read_csv(config.TEST)
X_test = X_test[X_test.columns[1:]]
X_test = scaler.transform(X_test)

submission = pd.read_csv(config.SUBMISSION)
submission.iloc[:, 1:] = model.predict_proba(X_test)
submission.to_csv('../../../../data/kaggle/TPS_JUNE2021/lgbm_baseline.csv', index=False)

