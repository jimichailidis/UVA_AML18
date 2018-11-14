from Blocks import Layer
import numpy as np

class MaxPool2x2(Layer):

    def forward(self, x_input):
        n_obj, n_ch, h, w = x_input.shape
        self.output = np.zeros((n_obj, n_ch, h // 2, w // 2))
        for i in range(n_obj):
            for j in range(n_ch):
                self.output[i, j] = maxpool_forward(x_input[i, j])
        return self.output

    def backward(self, x_input, grad_output):
        n_obj, n_ch, _, _ = x_input.shape
        grad_input = np.zeros_like(x_input)
        for i in range(n_obj):
            for j in range(n_ch):
                grad_input[i, j] = maxpool_grad_input(x_input[i, j], grad_output[i, j])
        return grad_input

def maxpool_forward(x_input):
    """Perform max pooling operation with 2x2 window
    # Arguments
        x_input: np.array of size (2 * W, 2 * H)
    # Output
        output: np.array of size (W, H)
    """

    W = x_input.shape[1]
    H = x_input.shape[0]
    stride = 2

    output_w = int((W-2)/stride + 1)
    output_h = int((H-2)/stride + 1)

    output = np.zeros((output_h, output_w))

    for w in range(0, W, stride):
        for h in range(0, H, stride):
            output[int(h/stride), int(w/stride)] = np.amax(x_input[h: h + stride, w: w + stride])


    return output

def maxpool_grad_input(x_input, grad_output):
    """Calculate partial derivative of the loss with respect to the input
    # Arguments
        x_input: np.array of size (2 * W, 2 * H)
        grad_output: partial derivative of the loss
            with respect to the output
            np.array of size (W, H)
    # Output
        output: partial derivative of the loss
            with respect to the input
            np.array of size (2 * W, 2 * H)
    """
    height, width = x_input.shape
    # Create an array of zeros the same size as the input
    grad_input = np.zeros(x_input.shape)

    # We set 1 if the element is the maximum in its window
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            window = x_input[i:i+2, j:j+2]
            i_max, j_max = np.unravel_index(np.argmax(window), (2, 2))
            grad_input[i + i_max, j + j_max] = 1

    # Overwrite with the corresponding gradient instead of 1
    grad_input = grad_input.ravel()
    grad_input[grad_input == 1] = grad_output.ravel()
    grad_input = grad_input.reshape(x_input.shape)
    return grad_input