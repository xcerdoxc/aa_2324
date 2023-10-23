import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler


# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def kernel_lineal(x1, x2):
    return x1.dot(x2.T)


# A scikit el kernel polinòmic tè per defecte:
# grau (degree) = 3
def kernel_poly(x1, x2, degree=3, sigma = 1/2):
    gamma = 1 / (2 * sigma ** 2)
    return (gamma*x1.dot(x2.T))**degree


def kernel_gauss(x1, x2, sigma=1/2):
    gamma = -1 / (2 * sigma ** 2)
    return np.exp(gamma * distance_matrix(x1, x2)**2)  #

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = MinMaxScaler()  #StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# Feim proves amb les dades i els kernels implementats
kernels = {"linear": kernel_lineal, "rbf": kernel_gauss, "poly": kernel_poly}

for kernel in kernels.keys():

    print(f"TEST KERNEL: {kernel}")
    svm_meu = SVC(C=1.0, kernel=kernels[kernel], random_state=33)
    svm_meu.fit(X_transformed, y_train)
    y_meu = svm_meu.predict(X_test_transformed)

    svm = SVC(C=1.0, kernel=kernel, random_state=33)
    svm.fit(X_transformed, y_train)
    y_scikit = svm.predict(X_test_transformed)


    print(" Resultats:")
    precision_scikit = precision_score(y_test, y_scikit)
    print(f"   Precisió Scikit: {precision_scikit}")
    precision_meu = precision_score(y_test, y_meu)
    print(f"   Precisió UIB   : {precision_meu}")

#  Exercici final
print("Exercici final: Canviam la dimensionalitat de les mostres i empram un classificador lineal")
poly = PolynomialFeatures(3)
X_poly = poly.fit_transform(X_transformed)
X_test_poly = poly.transform(X_test_transformed)
print(f"TEST features grau 3: {kernel}")
svm_poly = SVC(C=1.0, kernel="linear", random_state=33)
svm_poly.fit(X_poly, y_train)
y_meu = svm_poly.predict(X_test_poly)
precision_poly = precision_score(y_test, y_meu)
print(f"    Precisió   : {precision_poly}")

