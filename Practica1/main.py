from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd

train_path = 'dat/fashion-mnist_train.csv'
test_path = 'dat/fashion-mnist_test.csv'

# Dades entrenament
ds_train = pd.read_csv(train_path)

x_train = ds_train.iloc[:, 1:]  # Píxeles imágenes
y_train = ds_train.iloc[:, 0]  # Labels en la primera columna

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Dades test
ds_test = pd.read_csv(train_path)

x_test = ds_test.iloc[:, 1:]  # Píxeles imágenes
y_test = ds_test.iloc[:, 0]  # Labels en la primera columna

x_test_scaled = scaler.fit_transform(x_test)

param_grid = {
    'C': [0.1, 1, 10],             # Regularization parameter
    'kernel': ['linear', 'rbf'],    # Kernel type
    'gamma': [0.001, 0.01, 0.1]    # Kernel coefficient
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(x_train_scaled, y_train)

# Todo el dataset con el mejor estimador
best_svm_model = grid_search.best_estimator_
best_svm_model.fit(x_train_scaled, y_train)

# Predicción
y_pred = best_svm_model.predict(x_test)

# Avaluacio
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)