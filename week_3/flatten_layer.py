from Blocks import Layer

def flatten_forward(x_input):
    """Perform the reshaping of the tensor of size `(K, L, M, N)`
        to the tensor of size `(K, L*M*N)`
    # Arguments
        x_input: np.array of size `(K, L, M, N)`
    # Output
        output: np.array of size `(K, L*M*N)`
    """
    K, L, M, N = x_input.shape

    output = x_input.reshape(K, L*M*N)
    return output

def flatten_grad_input(x_input, grad_output):
    """Calculate partial derivative of the loss with respect to the input
    # Arguments
        x_input: partial derivative of the loss
            with respect to the output
            np.array of size `(K, L*M*N)`
    # Output
        output: partial derivative of the loss
            with respect to the input
            np.array of size `(K, L, M, N)`
    """
    K, L, M, N = x_input.shape
    grad_input = grad_output.reshape(K, L, M, N)
    return grad_input

class FlattenLayer(Layer):

    def forward(self, x_input):
        self.output = flatten_forward(x_input)
        return self.output

    def backward(self, x_input, grad_output):
        output = flatten_grad_input(x_input, grad_output)
        return output

