from classes.layers.convolution.Kernel import Kernel
from classes.layers.convolution.snippet import *

class Filter:

    def __init__(self, shape, bias=None, padding=1, strides=1):
        self.padding = padding
        self.strides = strides

        # Comprobando si indicaron el bias a utilizar
        if bias == None:
            bias = np.random.rand()
        self.bias = bias
        #self.bias = 0

        if type(shape) == tuple:  # (n_rows, n_cols, n_channels)
            self.kernels = [Kernel(shape[:2], self.padding, self.strides) for i in range(shape[2])]
        else:  # Si me indican los kernels especificos a usar
            self.kernels = shape

    def crossCorrelate(self, channels_tensor):
        """
        Función que calcula la cross correlacion de un filtro sobre un conjunto de imagenes
        :param channels_tensor: tiene la forma (n_instances, n_rows, n_cols, n_channels)
        """
        assert channels_tensor.shape[3] == len(self.kernels)  # Tienen que haber tantos filtros como canales

        # identificando la forma del tensor de salida
        individual_channel_shape = channelConvolutionDim(self.kernels[0].value.shape, channels_tensor[0, :, :, 0].shape,
                                                         self.padding, self.strides)

        output_shape = (channels_tensor.shape[0], individual_channel_shape[0], individual_channel_shape[1])
        output = np.zeros(output_shape)  # construyendo el tensor de salida
        output += self.bias  # Adicionando el bias a cada elemento del tensor

        # Calculando la suma de los mapas de caracteristicas y el bias
        for i in range(len(self.kernels)):
            channel_tensor = channels_tensor[:, :, :, i]
            output += self.kernels[i].crossCorrelate(channel_tensor)

        return output

    def convolve(self, channels_tensor):
        """
        Función que calcula la convolucion de un filtro sobre un conjunto de imagenes
        :param channels_tensor: tiene la forma (n_instances, n_rows, n_cols, n_channels)
        """
        assert channels_tensor.shape[3] == len(self.kernels)

        individual_channel_shape = channelConvolutionDim(self.kernels[0].value.shape, channels_tensor[0, :, :, 0].shape,
                                                         self.padding, self.strides)

        output_shape = (channels_tensor.shape[0], individual_channel_shape[0], individual_channel_shape[1])
        output = np.zeros(output_shape)
        output += self.bias

        for i in range(len(self.kernels)):
            channel_tensor = channels_tensor[:, :, :, i]
            output += self.kernels[i].convolve(channel_tensor)

        return output

    def __str__(self):
        return f"Filter(bias: {self.bias}, padding: {self.padding}, strides: {self.strides})"