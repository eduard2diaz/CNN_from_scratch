from classes.layers.Layer import Layer
from classes.layers.convolution.Filter import Filter
from classes.layers.convolution.snippet import *

class Convolution(Layer):

    def __init__(self, n_filters, filters_shape, padding=0, strides=1, activation_function=None):
        """
        :param input_shape: forma de los datos de entrada (n_instancias, rows, cols)
        :param n_filters: numero de filtros en la capa
        :param filters_shape: forma de los filtros (n_rows, n_cols, n_channels), de donde n_channels es el numero de
         kernels por filtro y (n_rows, n_cols) es la forma de cada uno de los kernels que componen el filtro. Por otra
         parte, si filters_shape no es una tupla refiere a los kernels por defecto a utilizar.
        :param padding: padding a aplicar sobre la matriz que sera cross_correlacionada/convolucionada
        :param strides: desplazamiento de cada kernel sobre la matrix a cross correlacionar/convolucionar
        :param activation_function: tupla con la funcion de activacion y su derivada a aplicar sobre la
         salida de cada filtro
        """

        # En las CNN los kernels son inicializados de forma aleatoria.
        self.filters = [Filter(filters_shape, padding=padding, strides=strides) for i in range(n_filters)]
        self.padding = padding
        self.strides = strides
        self.filters_shape = filters_shape
        self.activation_function = activation_function

    def forward(self, channels_tensor):
        """
        Función que calcula el forward (propagacion) de la entrada sobre los filtros de cross correlacion/convolucion
        :param channels_tensor: tiene la forma (n_instances, n_rows, n_cols, n_channels)
        """
        assert channels_tensor.shape[3] == len(self.filters[0].kernels)

        self.input = channels_tensor  # Guardamos los datos con que se entreno la capa
        # Calculamos las dimensiones resultantes de la cross correlacion
        individual_channel_shape = channelConvolutionDim(self.filters[0].kernels[0].value.shape,
                                                         channels_tensor[0, :, :, 0].shape, self.padding, self.strides)
        # Obteniendo la forma del tensor de salida
        output_shape = (channels_tensor.shape[0], individual_channel_shape[0],
                        individual_channel_shape[1], len(self.filters))
        output = np.zeros(output_shape)  # Inicializando el tensor de salida como todos ceros

        # Calculamos la cross correlacion de cada filtro
        for i in range(len(self.filters)):
            output[:, :, :, i] = self.filters[i].crossCorrelate(channels_tensor)

        if self.activation_function != None:  # Evaluacion de la funcion de activacion
            output = self.activation_function[0](output)

        return output

    def backward(self, output_gradient, learning_rate):
        """
        Función que calcula el backward (retropropagacion) de la capa
        :param output_gradient: gradiente de salida de la siguiente capa
        :param lr: factor de aprendizaje
        """
        # if self.activation_function != None:
        #    output_gradient = self.activation_function[1](output_gradient)

        n_instances, n_filters, n_channels = self.input.shape[0], len(self.filters), self.input.shape[3]

        # ---------------CALCULO DE LA DERIVADA DEL ERROR CON RESPECTO A LOS KERNELS-------------
        # definimos una mtz de gradientes para los kernels en los filtros
        kernels_gradient = np.zeros((n_filters, self.filters_shape[0], self.filters_shape[1], n_channels))
        # definimos una mtz de gradientes de la entrada
        input_gradient = np.zeros(self.input.shape)

        for instance_index in range(n_instances):
            instance = self.input[instance_index]  # Obteniendo la imagen
            for filter_index in range(n_filters):
                # Gradiente del filtro para la instancia
                filter_instance_grad = output_gradient[instance_index, :, :, filter_index]
                filter_kernels = self.filters[filter_index].kernels  # Obtengo los kernels del filtro
                for channel_index in range(n_channels):
                    # Calculamos la derivada del gradiente de salida con respecto a los kernels
                    # dE/dK_{ij} = Xj * dE/Yi
                    # Se uso un padding valid
                    kernel_gradient = crossCorrelation2D(instance[:, :, channel_index],
                                                             filter_instance_grad,
                                                             padding = self.padding)

                    kernels_gradient[filter_index, :, :, channel_index] += kernel_gradient

                    # Calculo del gradiente respecto a la entrada. En este caso usamos padding full
                    kernel_obj = filter_kernels[channel_index].value
                    channel_gradient = convolve2D(filter_instance_grad, kernel_obj, padding = kernel_obj.shape[0] - 1)
                    input_gradient[instance_index,:,:,channel_index] += channel_gradient

        # -------------------ACTUALIZACION DE LOS KERNELS
        for filter_index in range(n_filters):
            for channel_index in range(n_channels):
                kernel_dfilt_value = kernels_gradient[filter_index,:,:, channel_index]
                self.filters[filter_index].kernels[channel_index].value -= learning_rate * kernel_dfilt_value

        # ------------------ACTUALIZACION DE LOS BIAS
        # Sabemos que la derivada del error respecto a los bias del i-esimo filtro es igual a la derivada
        # del error respecto al gradiente del i-esimo filtro. Es decir, dE/dB_{i} = dE/Y_{i}. Por lo que:
        for filter_index in range(n_filters):
            self.filters[filter_index].bias = -learning_rate * np.average(
                output_gradient[:, :, :, filter_index].reshape((1, -1)))

        return input_gradient

    def __str__(self):
        return f"ConvLayer(n_filters: {len(self.filters)}, filters_shape: {self.filters_shape}, padding: {self.padding}, strides: {self.strides})"