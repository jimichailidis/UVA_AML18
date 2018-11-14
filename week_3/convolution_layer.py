from Blocks import Layer
import numpy as np

class ConvLayer(Layer):
    """
    Convolutional Layer. The implementation is based on
        the representation of the convolution as matrix multiplication
    """

    def __init__(self, n_in, n_out, filter_size):
        super(ConvLayer, self).__init__()
        self.W = np.random.normal(size=(n_out, n_in, filter_size, filter_size))
        self.b = np.zeros(n_out)


    def forward(self, x_input):
        n_obj, n_in, h, w = x_input.shape
        n_out = len(self.W)

        self.output = []

        for image in x_input:
            output_image = []
            for i in range(n_out):
                out_channel = 0.0
                for j in range(n_in):
                    out_channel += conv_matrix(image[j], self.W[i, j])
                output_image.append(out_channel)
            self.output.append(np.stack(output_image, 0))

        self.output = np.stack(self.output, 0)
        return self.output


    def backward(self, x_input, grad_output):

        N, C, H, W = x_input.shape
        F, C, HH, WW = self.W.shape

        pad = int((HH - 1) / 2)

        self.grad_b = np.sum(grad_output, (0, 2, 3))

        # pad input array
        x_padded = np.pad(x_input, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant')
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        # naive implementation of im2col
        x_cols = None
        for i in range(HH, H_padded + 1):
            for j in range(WW, W_padded+1):
                for n in range(N):
                    field = x_padded[n, :, i-HH:i, j-WW:j].reshape((1,-1))
                    if x_cols is None:
                        x_cols = field
                    else:
                        x_cols = np.vstack((x_cols, field))

        x_cols = x_cols.T

        d_out = grad_output.transpose(1, 2, 3, 0)
        dout_cols = d_out.reshape(F, -1)

        dw_cols = np.dot(dout_cols, x_cols.T)
        self.grad_W = dw_cols.reshape(F, C, HH, WW)

        w_cols = self.W.reshape(F, -1)
        dx_cols = np.dot(w_cols.T, dout_cols)

        dx_padded = np.zeros((N, C, H_padded, W_padded))
        idx = 0
        for i in range(HH, H_padded + 1):
            for j in range(WW, W_padded + 1):
                for n in range(N):
                    dx_padded[n:n+1, :, i-HH:i, j-WW:j] += dx_cols[:, idx].reshape((1, C, HH, WW))
                    idx += 1
            dx = dx_padded[:, :, pad:-pad, pad:-pad]
        grad_input = dx
        return grad_input


    def get_params(self):
        return [self.W, self.b]


    def get_params_gradients(self):
        return [self.grad_W, self.grad_b]


def conv_matrix(matrix, kernel):
    """
    Perform the convolution of the matrix
    with the kernel using zero padding
    # Arguments
        matrix: input matrix np.array of size `(N, M)`
        kernel: kernel of the convolution
            np.array of size `(2p + 1, 2q + 1)`
    # Output
        the result of the convolution
        np.array of size `(N, M)`
    """

    q = int((kernel.shape[1] - 1)/2)
    p = int((kernel.shape[0] - 1)/2)

    output = np.zeros_like(matrix)
    matrix = np.pad(matrix, [(p, p), (q, q)], mode='constant', constant_values=0)

    for i in range(0, output.shape[0], 1):
        for j in range(0, output.shape[1], 1):
            output[i, j] = np.sum(kernel * matrix[i:i+2*p+1, j:j+2*q+1])

    return output
