# neural-network-basics

## binary-classification

binary classification is used to classify data into two categories

outputs are either 0 or 1

- identify cat in image exemple

    image is composed of 3 matrices of pixels (red, green, blue) with numbers between 0 and 255

    so x can be an array of the values of the tree matrices (3, 64, 64) flattened to (12288, 1)

    and y is a 0 (not a cat) or 1 (is a cat)

X is used to denote the whole imput data set and Y all the coresponding labels

``` python
# m is the number of exemples
# n is the number of features
X.shape # (m , n)
Y.shape # (1, m)
```

### logistic-regression

given x, we want to predict ŷ = P(y=1|x)

we use the sigmoid function to map the input to a probability

ŷ = sigmoid(wT*x + b)

### loss-function

the loss function calculate the error between the prediction and the real value over one exemple

L(ŷ, y) = -ylog(ŷ) - (1-y)log(1-ŷ)

### cost-function

the cost function is the average of the loss function over all the exemples

J(w, b) = 1/m * sum(L(ŷ, y))

### gradient-descent

we want to find the best parameters w and b so we use gradient descent to find the minimum of the cost function

- start with random w and b

- iterate until convergence

    - w := w - alpha * dJ(w, b)/dw

    - b := b - alpha * dJ(w, b)/db

alpha is the learning rate 

``` python
dJ_dw[j] = 1/m * x[i][j] * sum(y_hat[i] - y[i])
dJ_db = 1/m * sum(y_hat[i] - y[i])
```

### vectorized-implementation

``` python
import numpy as np
# m is the number of exemples
# n is the number of features
X.shape # (m , n)
Y.shape # (1, m)
W.shape # (n, 1)
b.shape # (1, 1)

Z = np.dot(W.T, X) + b
A = 1/(1 + np.exp(-Z))
W -= alpha * (1/m) * np.dot(X, (A - Y).T)
b -= alpha * (1/m) * np.sum(A - Y)
```

