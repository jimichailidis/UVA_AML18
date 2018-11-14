import numpy as np

def dense_forward(x_input, W, b):
    """Perform the mapping of the input
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the output of a dense layer 
        np.array of size `(n_objects, b_out)`
    """
    
    output = np.matmul(x_input, W) + b
    return output


def dense_grad_W(x_input, grad_output, W, b):
    """Calculate the partial derivative of 
        the loss with respect to W parameter of the layer
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dense layer 
            np.array of size `(n_objects, n_out)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the partial derivative of the loss 
        with respect to W parameter of the layer
        np.array of size `(n_in, n_out)`
    """
    
    grad_W = x_input.T.dot(grad_output)
    return grad_W

def dense_grad_input(x_input, grad_output, W, b):
    """Calculate the partial derivative of 
        the loss with respect to the input of the layer
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dense layer 
            np.array of size `(n_objects, n_out)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the partial derivative of the loss with
        respect to the input of the layer
        np.array of size `(n_objects, n_in)`
    """
    grad_input = grad_output.dot(W.T)
    
    return grad_input


def dense_grad_b(x_input, grad_output, W, b):
    """Calculate the partial derivative of 
        the loss with respect to b parameter of the layer
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dense layer 
            np.array of size `(n_objects, n_out)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the partial derivative of the loss 
        with respect to b parameter of the layer
        np.array of size `(n_out,)`
    """
    
    grad_b = grad_output.sum(axis=0)
    return grad_b


def relu_forward(x_input):
    """relu nonlinearity
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
    # Output
        the output of relu layer
        np.array of size `(n_objects, n_in)`
    """
    output = np.maximum(np.zeros(x_input.shape), x_input)
    return output

def relu_grad_input(x_input, grad_output):
    """relu nonlinearity gradient. 
        Calculate the partial derivative of the loss 
        with respect to the input of the layer
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
            grad_output: np.array of size `(n_objects, n_in)`
    # Output
        the partial derivative of the loss 
        with respect to the input of the layer
        np.array of size `(n_objects, n_in)`
    """
    relu = relu_forward(x_input)
    # Derivative
    relu[relu > 0] = 1
    grad_input = relu * grad_output
    return grad_input


def hinge_forward(target_pred, target_true):
    """Compute the value of Hinge loss 
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the value of Hinge loss 
        for a given prediction and the ground truth
        scalar
    """
    
    error_sum = sum(np.maximum(np.zeros_like(target_pred), 1 - target_pred * target_true))
    output = error_sum / len(target_pred)
    
    return output

def hinge_grad_input(target_pred, target_true):
    """Compute the partial derivative 
        of Hinge loss with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the partial derivative 
        of Hinge loss with respect to its input
        np.array of size `(n_objects,)`
    """
    test = target_pred * target_true
    mask = test < 1
    
    new_1 = -target_true[mask]/target_true.shape[0]
    new_2 = 0
    
    grad_input = np.empty_like(test)
    grad_input[mask] = new_1
    grad_input[~mask] = new_2

    return grad_input


def l2_regularizer(weight_decay, weights):
    """Compute the L2 regularization term
    # Arguments
        weight_decay: float
        weights: list of arrays of different shapes
    # Output
        sum of the L2 norms of the input weights
        scalar
    """
    
    l2_norm = np.linalg.norm(weights) ** 2
    output = weight_decay/2 * l2_norm
    
    return output


###################################################
#        put here whatever you want               #
###################################################


class Layer(object):

    def __init__(self):
        self.training_phase = True
        self.output = 0.0

    def forward(self, x_input):
        self.output = x_input
        return self.output

    def backward(self, x_input, grad_output):
        return grad_output

    def get_params(self):
        return []

    def get_params_gradients(self):
        return []


class Dense(Layer):

    def __init__(self, n_input, n_output):
        super(Dense, self).__init__()
        # Randomly initializing the weights from normal distribution
        self.W = np.random.normal(size=(n_input, n_output))
        self.grad_W = np.zeros_like(self.W)
        # initializing the bias with zero
        self.b = np.zeros(n_output)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x_input):
        self.output = dense_forward(x_input, self.W, self.b)
        return self.output

    def backward(self, x_input, grad_output):
        # get gradients of weights
        self.grad_W = dense_grad_W(x_input, grad_output, self.W, self.b)
        self.grad_b = dense_grad_b(x_input, grad_output, self.W, self.b)
        # propagate the gradient backwards
        return dense_grad_input(x_input, grad_output, self.W, self.b)

    def get_params(self):
        return [self.W, self.b]

    def get_params_gradients(self):
        return [self.grad_W, self.grad_b]


class ReLU(Layer):

    def forward(self, x_input):
        self.output = relu_forward(x_input)
        return self.output

    def backward(self, x_input, grad_output):
        return relu_grad_input(x_input, grad_output)


class SequentialNN(object):

    def __init__(self):
        self.layers = []
        self.training_phase = True

    def set_training_phase(self, is_training=True):
        self.training_phase = is_training
        for layer in self.layers:
            layer.training_phase = is_training

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x_input):
        self.output = x_input
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output

    def backward(self, x_input, grad_output):
        inputs = [x_input] + [l.output for l in self.layers[:-1]]
        for input_, layer_ in zip(inputs[::-1], self.layers[::-1]):
            grad_output = layer_.backward(input_, grad_output)

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def get_params_gradients(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_params_gradients())
        return grads


class Loss(object):

    def __init__(self):
        self.output = 0.0

    def forward(self, target_pred, target_true):
        return self.output

    def backward(self, target_pred, target_true):
        return np.zeros_like(target_pred)


class Hinge(Loss):

    def forward(self, target_pred, target_true):
        self.output = hinge_forward(target_pred, target_true)
        return self.output

    def backward(self, target_pred, target_true):
        return hinge_grad_input(target_pred, target_true)


class Optimizer(object):
    '''This is a basic class. 
    All other optimizers will inherit it
    '''

    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def update_params(self):
        pass


class SGD(Optimizer):
    '''Stochastic gradient descent optimizer
    https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    '''

    def update_params(self):
        weights = self.model.get_params()
        grads = self.model.get_params_gradients()
        for w, dw in zip(weights, grads):
            update = self.lr * dw + self.weight_decay * w
            # it writes the result to the previous variable instead of copying
            np.subtract(w, update, out=w)