from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

def kernel_lineal(x1, x2):
    return x1.dot(x2.T)
def kernel_gaussia(x, z, gamma=.01):
    norm_squared = distance_matrix(x, z)
    return np.exp(-gamma * norm_squared)



X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# En realitat ja no necessitem canviar les etiquetes Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler() #StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)
res=kernel_gaussia(X_transformed, y_train)


#Entrenam una SVM linear (classe SVC)
svm = SVC(C=1.0, kernel=kernel_lineal)
svm.fit(X_transformed, y_train)
y_predicted = svm.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)

print(f'Rati d\'acerts en el bloc de predicció: {(len(y_predicted)-errors)/len(y_predicted)}')

svm = SVC(C=1.0, kernel=kernel_gaussia)
svm.fit(X_transformed, y_train)
y_predicted = svm.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)

print(f'Rati d\'acerts en el bloc de predicció: {(len(y_predicted)-errors)/len(y_predicted)}')