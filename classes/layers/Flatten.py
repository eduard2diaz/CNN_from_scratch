from classes.layers.Layer import *

class Flatten(Layer):

    def forward(self, mtx):
        """
        El metodo forward convierte a vector fila la entrada.
        :param mtx: Salida del forward de la capa anterior. Tiene la forma (n_instances, n_rows, n_cols, n_filters)
        :return: numpy.ndarray
        """
        self.input_shape = mtx.shape
        return mtx.reshape(mtx.shape[0], -1)

    def backward(self, output_gradient, learning_rate):
        """
        EL metodo backward en la clase Flatten toma el gradiente de salida que recibio del backward de la siguiente
        capa y le cambia la forma, para que tenga las mismas dimensiones que tenia la entrada que habia recibido
        la clase Flatten en el metodo forward.
        :param output_gradient: gradiente de salida arrojado por el metodo forward de la siguiente capa
        :param learning_rate: factor de aprendizaje
        :return: numpy.ndarray
        """
        return output_gradient.reshape(self.input_shape)

    def __str__(self):
        return f"FlattenLayer()"