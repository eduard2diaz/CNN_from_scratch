from classes.layers.Layer import Layer
from classes.layers.convolution.snippet import *

class AveragePooling(Layer):
    def __init__(self, pooling_shape, padding=0, strides=2):
        # El filtro de pooling debe considerar al menos 4 elementos
        assert pooling_shape[0] > 1 and pooling_shape[1] > 1
        self.pooling_shape = pooling_shape
        self.padding = padding
        self.strides = strides
        self.last_input = None

    def forward(self, mtx_tensor):  # mtx_tensor : (n_instances, n_rows, n_cols, n_filters)
        self.last_input = mtx_tensor

        mtx_instances, mtx_rows, mtx_cols, mtx_filters = mtx_tensor.shape
        # Calculando la forma del tensor de salida
        output_pooling_shape = channelConvolutionDim(self.pooling_shape, (mtx_rows, mtx_cols),
                                                     self.padding, self.strides)
        # Inicializando el tensor de salida
        output = np.zeros((mtx_instances, output_pooling_shape[0], output_pooling_shape[1], mtx_filters))

        def averagePoolingSelection(mtx, output):
            """
            Funcion que calcula maxPooling para una matriz de la forma (n_rows, n_cols). Es decir, crea una submatriz
            compuesta por el valor promedio de cada uno de los "cuadrantes"
            :param mtx: matriz sobre la que calcular averagePooling
            :param padding: padding a utilizar
            :param strides: strides a utilizar
            :param output: puntero a la matriz resultante
            """
            if self.padding > 0:
                mtx_padding = np.zeros((2 * self.padding + mtx.shape[0], 2 * self.padding + mtx.shape[1]))
                mtx_padding[self.padding: mtx.shape[0] + self.padding, self.padding: mtx.shape[1] + self.padding] = mtx
                mtx = mtx_padding

            mtx_i = 0
            for i in range(output.shape[0]):
                mtx_j = 0
                for j in range(output.shape[1]):
                    submatrix = mtx[mtx_i: mtx_i + self.pooling_shape[0],
                                mtx_j: mtx_j + self.pooling_shape[1]]
                    output[i][j] = np.average(submatrix.reshape((-1, 1)))
                    mtx_j += self.strides
                mtx_i += self.strides

        for instance in range(mtx_instances):
            for channel in range(mtx_filters):
                averagePoolingSelection(mtx_tensor[instance, :, :, channel], output[instance, :, :, channel])

        return output

    def backward(self, output_gradient, learning_rate):
        """
        Funcion que calcula la derivada de la salida respecto a las entradas de la capa
        :param output_gradient: gradiente de salida de la siguiente capa.
        """
        # Recreando la matriz a la que se aplico maxPooling
        output = np.zeros(self.last_input.shape)
        mtx_instances, mtx_rows, mtx_cols, mtx_filters = output.shape

        def averagePoolingSelectionRevert(last_input_mtx, output_gradient):
            if self.padding > 0:
                mtx_padding = np.zeros((2 * self.padding + last_input_mtx.shape[0],
                                        2 * self.padding + last_input_mtx.shape[1]))

                mtx_padding[self.padding: last_input_mtx.shape[0] + self.padding,
                self.padding: last_input_mtx.shape[1] + self.padding] = last_input_mtx

                last_input_mtx = mtx_padding

            # Creo una mtz de ceros con la misma forma de la mtz de entrada ya con padding, donde almacenare la suma de los
            # gradientes de entrada
            padding_out = np.zeros(last_input_mtx.shape)

            mtx_i = 0
            for i in range(output_gradient.shape[0]):
                mtx_j = 0
                for j in range(output_gradient.shape[1]):
                    gradiente = output_gradient[i][j]  # obtengo el gradiente
                    # Actualizo la matriz de gradientes de entrada
                    gradiente = gradiente/(self.pooling_shape[0] * self.pooling_shape[1])
                    #Divido el gradiente entre el numero de instancias que se consideraron en el pooling
                    #y este resultado es sumado a los gradientes de cada una de las instancias de dichas
                    #instancias.
                    padding_out[mtx_i: mtx_i + self.pooling_shape[0],
                    mtx_j: mtx_j + self.pooling_shape[1]] += gradiente
                    mtx_j += self.strides
                mtx_i += self.strides

            return padding_out[self.padding: self.last_input.shape[1] + self.padding,
                   self.padding: self.last_input.shape[2] + self.padding]

        for instance in range(mtx_instances):
            for filter_index in range(mtx_filters):
                output[instance, :, :, filter_index] = averagePoolingSelectionRevert(self.last_input[instance, :, :, filter_index],
                                                                        output_gradient[instance, :, :, filter_index])

        return output

    def __str__(self):
        return f"AveragePoolingLayer(padding: {self.padding}, strides: {self.strides})"