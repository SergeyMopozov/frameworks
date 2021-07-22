import config
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


print("[INFO] loading data...")
dataset = pd.read_csv(config.TRAIN)
X = dataset[dataset.columns[1:-1]]
y = dataset[config.TARGET]
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, random_state=0, test_size=0.25)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# set param grid and model

# Create parameters to search
grid = {
    'learning_rate': [0.005],
    'n_estimators': [40, 200],
    'num_leaves': [6, 8, 12, 16],
    'colsample_bytree': [0.65, 0.66],
    'subsample': [0.7, 0.75],
    'reg_alpha': [1, 1.2],
    'reg_lambda': [1, 1.2, 1.4],
    }

model = LGBMClassifier(random_state=0)

# initialize a cross-validation fold and perform a grid-search to
# tune the hyperparameters
print("[INFO] grid searching over the hyperparameters...")
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cvFold, scoring='neg_log_loss')
searchResults = gridSearch.fit(X_train, y_train)

# extract the best model and evaluate it
bestModel = searchResults.best_estimator_

# evaluate our model using logloss-score
print("[INFO] evaluating...")
loss = log_loss(pd.get_dummies(y_train).values, bestModel.predict_proba(X_train))
print(f"Train Log Loss: {loss:.5f}")
loss = log_loss(pd.get_dummies(y_valid).values, bestModel.predict_proba(X_valid))
print(f"Valid Log Loss: {loss:.5f}")


print("[INFO] make submission")
X_test = pd.read_csv(config.TEST)
X_test = X_test[X_test.columns[1:]]
X_test = scaler.transform(X_test)

submission = pd.read_csv(config.SUBMISSION)
submission.iloc[:, 1:] = bestModel.predict_proba(X_test)
submission.to_csv('../../../../data/kaggle/TPS_JUNE2021/lgbm_gscv.csv', index=False)