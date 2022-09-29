from classes.layers.convolution.snippet import *

class Kernel:

    def __init__(self, shape, padding=1, strides=1):  # shape: (n_rows, n_cols)
        self.padding = padding
        self.strides = strides
        # En python no existe la sobrecarga de metodos!!!
        if type(shape) == tuple:
            self.value = np.random.rand(shape[0], shape[1]) * 2 - 1
        else:
            self.value = shape  # Specifing the kernel

    def crossCorrelate(self, channel_tensor):
        """
        Funcion para el calculo de la coorelacion cruzada
        :param channel_tensor: es un tensor con los datos de un determinado canal para diferentes instancias/imagenes,
        tiene la forma (n_instances, n_rows, n_cols)
        """
        return np.array([crossCorrelation2D(image, self.value, self.padding, self.strides) for image in channel_tensor])

    def convolve(self, channel_tensor):
        """
        Funcion para el calculo de la convolucion, por lo que dentro de la misma se rotara 180 grados el filtro
        :param channel_tensor: es un tensor con los datos de un determinado canal para diferentes instancias/imagenes,
        tiene la forma (n_instances, n_rows, n_cols)
        """
        return np.array([convolve2D(image, self.value, self.padding, self.strides) for image in channel_tensor])

    def __str__(self):
        return f"Kernel(padding: {self.padding}, strides: {self.strides})"