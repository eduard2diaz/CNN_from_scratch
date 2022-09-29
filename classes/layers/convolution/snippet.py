import numpy as np

def channelConvolutionDim(kernel_shape, image_channel_shape, padding=0, strides=1):
    """
    Funcionalidad que retorna las dimensiones de salida resultantes de aplicar cross correlacion/convolucion.
    :param kernel_shape: es la forma del kernel de cross_correlacion/convolucion. Tiene la forma (width, height)
    :param image_channel_shape: forma de la matriz sobre la que aplicar el kernel. El 1er parametro es el ancho
    y el 2do el alto.
    :param padding: padding a aplicar sobre la matriz que sera cross_correlacionada/convolucionada
    :param strides: desplazamiento del kernel sobre la matriz
    """   
    # Gather Shapes of Kernel + Image + Padding
    (xKernShape, yKernShape) = kernel_shape
    (xImgShape, yImgShape) = image_channel_shape

    # Shape of Output Convolution(Se multiplica por 2 pues el padding es agregado en cada uno de extremos del eje)
    #(output_shape =  [[W - K + 2P]/S] + 1)
    #donde W es la la forma de la entrada, K la forma del kernel, P es el padding y S el stride 
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1) 
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    return (xOutput, yOutput)

def getIterableZone(image, kernel_shape, strides = 1):
    """
    Funcionalidad que dada una imagen retorna cada uno de los 'cuadrantes' a procesar por el kernel
    :param image: imagen a procesar
    :param kernel_shape: forma del kernel (nrows, ncols)
    :param strides: desplazamiento dentro de la imagen
    """
    n_rows, n_cols = channelConvolutionDim(kernel_shape, image.shape, strides = strides)
    x = 0
    for i in range(n_rows):
        y = 0
        for j in range(n_cols):
            x_end = x+kernel_shape[0] 
            y_end = y+kernel_shape[1]
            im_region = image[x: x_end, y: y_end]
            yield im_region, i, j
            y+=strides
        x+=strides        

def crossCorrelation2D(image, kernel, padding=0, strides=1):
    """
    Funcionalidad que realiza la cross correlacion de un kernel sobre una imagen
    :param image: imagen a procesar
    :param kernel: kernel a utilizar
    :param padding: numero de filas y columas de cero a agregar en los extremos
    :param strides: desplazamiento dentro de la imagen
    """
    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image
    
    it = 0
    for zone, i, j in getIterableZone(imagePadded, kernel.shape, strides):
        row = it//yOutput
        col = it - row * yOutput
        temp = zone * kernel
        output[row][col] = np.sum(np.array(temp).reshape((1,-1)))
        it+=1

    return output

def convolve2D(image, kernel, padding=0, strides=1):
    """
    Funcionalidad que realiza la convolucion de un kernel sobre una imagen
    :param image: imagen a procesar
    :param kernel: kernel a utilizar
    :param padding: numero de filas y columas de cero a agregar en los extremos
    :param strides: desplazamiento dentro de la imagen
    """
    #Flip the kernel 180 degree
    kernel = np.flipud(np.fliplr(kernel))
    # Cross Correlation
    return crossCorrelation2D(image, kernel, padding, strides)