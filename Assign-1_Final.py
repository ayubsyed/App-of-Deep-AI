'''This task creates a 3-layer neural network using only NumPy'''
import numpy as np

# Setting the random seed for consistent production of accuracy results
np.random.seed(300)

# Loads the dataset and splits into training and testing sets.
fname = 'assign1_data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

# Intializes the DenseLayer with weights and biases.
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

# Performs the forward pass (calculates the output of the layer) through the
# layer using the parameters.
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases

# Performs the backward pass (calculates the gradients with respect to the
# layer's parameters) through the layer.
    def backward(self, dz):
        self.dweights = np.dot(self.inputs.T, dz)
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        self.dinputs = np.dot(dz, self.weights.T)

# Utilizes the ReLU activation function
class ReLu:
    # Computes the ReLU activation for the input 'z'.
    def forward(self, z):
        self.z = z
        self.activity = np.maximum(0, z)

    # Computes the gradient of the ReLU function.
    def backward(self, dactivity):
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0

# Utilizes the Softmax activation function.
class Softmax:
    # Computes the Softmax activation for the input 'z'.
    def forward(self, z):
        # Computes the exponentials of 'z'.
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        # Normalizes the exponentials to obtain probabilities.
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs

# Computes the gradient of the Softmax function.
    def backward(self, dprobs):
        # Initializes an empty array for the gradients.
        self.dz = np.empty_like(dprobs)
        # Computes the gradient for each sample in the batch.
        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            # Reshapes the probabilities to be column vectors.
            prob = prob.reshape(-1, 1)
            # Computes the Jacobian matrix.
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            # Calculates the gradient of the loss with respect to 'z'.
            self.dz[i] = np.dot(jacobian, dprob)

# Utilizes the cross-entropy loss function.
class CrossEntropyLoss:
    # Calculates CrossEntropyLoss between predicted probabilities vs true labels.
    def forward(self, probs, oh_y_true):
        # Clips the probabilities to avoid division by zero or log of zero.
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        # Calculates the cross-entropy loss for each sample.
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return loss.mean(axis=0)

    # Calculates gradient of the loss with respect to predicted probabilities.
    def backward(self, probs, oh_y_true):
        # Gets the batch size and number of classes.
        batch_sz, n_class = probs.shape
        # Calculates the gradient of the loss.
        self.dprobs = -oh_y_true / probs
        # Normalize the gradient by dividing by the batch size.
        self.dprobs = self.dprobs / batch_sz

# Utilizes the Stochastic Gradient Descent (SGD) optimizer.
class SGD:
    def __init__(self, learning_rate=1.0):
        # Initializes the default learning rate for the optimizer.
        self.learning_rate = learning_rate

# Updates the parameters of a layer.
    def update_params(self, layer):
        # Updates the weights by subtracting their gradient.
        layer.weights -= self.learning_rate * layer.dweights
        # Updates the biases by subtracting the gradient of the biases.
        layer.biases -= self.learning_rate * layer.dbiases

# Generates predictions from the predicted probabilities.
def predictions(probs):
    y_preds = np.argmax(probs, axis=1)
    return y_preds

# Calculates the accuracy of predictions.
def accuracy(y_preds, y_true):
    return np.mean(y_preds == y_true)

# Refers to the cross-entropy loss function.
loss_function = CrossEntropyLoss()

# Performs a forward pass through the entire network.
def forward_pass(X, y_true, n_class=3):
    dense1.forward(X)
    activation1.forward(dense1.z)
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)
    output_layer.forward(activation2.activity)
    probs = output_activation.forward(output_layer.z)
    return probs

# Performs a backward pass through the entire network.
def backward_pass(probs, y_true, oh_y_true):
    loss_function.backward(probs, oh_y_true)
    output_activation.backward(loss_function.dprobs)
    output_layer.backward(output_activation.dz)
    activation2.backward(output_layer.dinputs)
    dense2.backward(activation2.dz)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dz)

# Defines the network architecture and traines hyperparameters.

# Number of input features.
n_inputs = 3
# Number of neurons in the first hidden layer.
n_hidden1 = 4
# Number of neurons in the second hidden layer.
n_hidden2 = 8
# Number of output classes.
n_outputs = 3
# Number of training epochs.
epochs = 9
# Batch size.
batch_size = 32

# Initializes the layers of the neural network.
dense1 = DenseLayer(n_inputs, n_hidden1)
activation1 = ReLu()

dense2 = DenseLayer(n_hidden1, n_hidden2)
activation2 = ReLu()

output_layer = DenseLayer(n_hidden2, n_outputs)
output_activation = Softmax()

# Instantiate the cross-entropy loss function.
crossentropy = CrossEntropyLoss()
# Initialize the optimizer.
optimizer = SGD()

# Training Loop.
for epoch in range(epochs):
    print('epoch:', epoch)
    # Calculates the number of batches.
    n_batch = len(X_train) // batch_size
    # Loops over each batch.
    for batch_i in range(n_batch):
        batch_X = X_train[batch_i * batch_size:(batch_i + 1) * batch_size]
        batch_y = y_train[batch_i * batch_size:(batch_i + 1) * batch_size]
        # One-hot encodes the labels.
        oh_batch_y = np.eye(n_outputs)[batch_y]
        # Performs a forward pass to compute the predicted probabilities.
        probs = forward_pass(batch_X, batch_y)
        # Calculates the cross-entropy loss between  predicted probabilities and
        # the one-hot encoded true labels
        loss = crossentropy.forward(probs, oh_batch_y)
        # Makes predictions based on the probabilities.
        y_preds = predictions(probs)
        # Calculates the accuracy of the current batch.
        batch_accuracy = accuracy(y_preds, batch_y)
        print(f'Batch {batch_i + 1}/{n_batch}, Loss: {loss:.4f}, \
        Accuracy: {batch_accuracy:.2f}%')
        # Performs a backward pass to compute the gradients.
        backward_pass(probs, batch_y, oh_batch_y)

        # Updates the parameters of each layer using the optimizer.
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(output_layer)

probs = forward_pass(X_test, y_test)
print(probs)
accuracy_of_test = accuracy(predictions(probs), y_test)
print(accuracy_of_test)