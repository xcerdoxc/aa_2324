from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd

train_path = 'dat/fashion-mnist_train.csv'
test_path = 'dat/fashion-mnist_test.csv'

# Dades entrenament
ds_train = pd.read_csv(train_path)

x_train = ds_train.iloc[:, 1:]  # Pixels imatges
y_train = ds_train.iloc[:, 0]  # Labels primera columna

# Dades test
ds_test = pd.read_csv(test_path)

x_test = ds_test.iloc[:, 1:]  # Pixels imatges
y_test = ds_test.iloc[:, 0]  # Labels primera columna

# Escalam
scaler = StandardScaler()
#scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    print(f"TEST KERNEL: {kernel}")
    svm = SVC(kernel=kernel)
    svm.fit(x_train_scaled, y_train)
    y_pred = svm.predict(x_test_scaled)
    # Avaluacio
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

#param_grid = {
#    'C': [0.1, 1, 10],             # Parametre de regularitzacio
#    'kernel': ['linear', 'rbf'],    # falta un kernel
#    'gamma': [0.001, 0.01, 0.1]    # Coeficient de Kernel
#}

#grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=1, n_jobs=-1)
#grid_search.fit(x_train_scaled, y_train)

# Dataset amb el millor estimador
#best_svm_model = grid_search.best_estimator_

#model_path = 'dat/best_svm_model.joblib'
#joblib.dump(best_svm_model, model_path)
#
#best_svm_model.fit(x_train_scaled, y_train)

# Prediccio
#y_pred = best_svm_model.predict(x_test)

#svm.fit(x_train_scaled, y_train)
#y_pred = svm.predict(x_test)
