# neural-network-overview

## representation

input layer -> hidden layer -> ... -> hidden layer -> output layer

each layer is composed of multiple neurons, each with an activation function

the output of each neuron in a layer is the input of each neuron in the next layer

$$ a^{[0]} = X = inputLayer $$

$$ a^{[1:-1]} = hidden layers $$

$$ a^{[-1]} = Yhat = outputLayer $$

for each hidden layer, the inputs are the output of the previous layer and the outputs are the sigmoid of the linear combination of the inputs and the weights plus the bias

supose we have a neural network with 3 layers, the input layer with 3 features, the hidden layer with 4 neurons and the output layer with 1 neuron

$$ w^{[i]}_j $$ is the weight of the j-th neuron in the i-th layer and its shape is (1, number of neurons in previous layer)

## vectorized-implementation

``` python
# X is the input layer
A[1] = activation(W[1]X + b[1])  # layer 1 (hidden layer)
A[2] = activation(W[2]A[1] + b[2])  # layer 2 (output layer)
```

## activation-function

tanh and sigmoid are common for classification problems

- sigmoid: $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

$$ D(\sigma) = (0, 1) $$

- tanh: $$ tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $$

$$ D(tanh) = (-1, 1) $$

- relu: $$ relu(z) = max(0, z) $$

$$ D(relu) = (0, \infty) $$

- linear: $$ linear(z) = z $$

$$ D(linear) = (-\infty, \infty) $$

for hidden layers the relu function is the most common

## gradient-descent

### derivatives

- derivative of sigmoid:

$$ \frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z)) $$

- derivative of tanh:

$$ \frac{d\tanh}{dz} = 1 - \tanh^2(z) $$

- derivative of relu:

$$ \frac{drelu}{dz} = \begin{cases} 0 & z < 0 \\ 1 & z \geq 0 \end{cases} $$

- derivative of linear:

$$ \frac{dlinear}{dz} = 1 $$

### update-parameters

$$ dw^{[i]} = \frac{dJ}{dw^{[i]}} $$

$$ db^{[i]} = \frac{dJ}{db^{[i]}} $$

$$ W^{[i]} := W^{[i]} - \alpha dw^{[i]} $$

$$ b^{[i]} := b^{[i]} - \alpha db^{[i]} $$

## random-initialization

initialize all weights to zero will cause all neurons to be the same

so all values must be random and small so that the derivative is not too small