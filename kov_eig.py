import numpy as np
from scipy.linalg import eig


T = np.mat("0.2 0.4 0.4;0.8 0.2 0.0;0.8 0.0 0.2")
print(T)

# returns eigenvalues, left eigenvetor, right eigenvectors in order
# eigen vectors are normalized
# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.linalg.eig.html
from scipy.linalg import eig
w, el, er = eig(T, left=True)
print(w)
print(el)
print(er)