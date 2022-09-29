from classes.layers.convolution.snippet import *
from classes.layers.Convolution import Convolution
from classes.layers.Flatten import Flatten
from classes.layers.convolution.Filter import *
import cv2 as cv #Modulo de trabajo con opencv

def read_image(image):
    """
    Funcionalidad que carga una imagen
    :param: direccion de la imagen a cargar
    """
    return cv.imread(image, cv.COLOR_BGR2RGB) #Leyendo la imagen en formato RGB

image = read_image('image2.jpeg')

import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()
print('Image shape', image.shape)

#Getting channels
(R,G,B) = cv.split(image)
#Printing channels shapes
print('R channel shape', R.shape)
print('G channel shape', G.shape)
print('B channel shape', B.shape)
#Saving channels (Each photo are in black and white colors because they has only one channel)
cv.imwrite("./channel_red.jpg", R)
cv.imwrite("./channel_green.jpg", G)
cv.imwrite("./channel_blue.jpg", B)

#Watching the channels through matplotlib
fig, axs = plt.subplots(1, 3, figsize = (20,5))
channels = ['Red', 'Green', 'Blue']

for i, ax in enumerate(axs.flat):
    ax.imshow(image[:,:,i], cmap ='gray')
    ax.set_title(f"Channel {i+1} : {channels[i]}")
plt.show()

#Convolución de matrices
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

output = convolve2D(R, edge_kernel, padding = 1)

plt.imshow(output, cmap = 'gray')
plt.show()
cv.imwrite('convolution_result.jpeg', output)
print('output.shape', output.shape)

k1 = Kernel(edge_kernel, padding = 1)
image_channels_array = np.array([image[:,:, channel] for channel in range(image.shape[2])])
print('image.shape', image.shape, 'image_channels_array.shape', image_channels_array.shape)

output2 = k1.convolve(image_channels_array)
print('output2.shape', output2.shape)

fig, axs = plt.subplots(1, 3, figsize = (20,5))

for i, ax in enumerate(axs.flat):
    ax.imshow(output2[i, :,:], cmap ='gray')
    ax.set_title(f"Convolution Channel {i+1} : {channels[i]}")
plt.show()

print(k1)

#Filtros de convolución y cross correlación

#Probando la convolución de varios kernels
#Definiremos tantos kernels como canales, cada uno de los cuales procesará un respectivo canal

kernels = [
    Kernel(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]), padding = 1),
    Kernel(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), padding = 1),
    Kernel(np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), padding = 1)
]

kernels_name = ['Ridge detection','Identity', 'Sharpen']

fig, axs = plt.subplots(1, 3, figsize = (20,5))

for i, ax in enumerate(axs.flat):
    image_array = np.array([image[:,:,i]])  #Seleccionando la capa a convolucionar
    #Graficando la convolucion
    conv_result = kernels[i].convolve(image_array)
    ax.imshow(conv_result[0], cmap ='gray')
    ax.set_title(f"Channel {channels[i]} -> Kernel {kernels_name[i]}")

plt.show()

#Creando un filtro de convolución y cross correlación
#Prueba 1
image_new = image.reshape((1,) + image.shape)
print('image_new.shape', image_new.shape)

kernels2 = [
    Kernel(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), padding = 1),
    Kernel(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), padding = 1),
    Kernel(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), padding = 1)
]

f1 = Filter(kernels2, bias = 0)
print(f"Filter Bias: {f1.bias}")

output3 = f1.convolve(image_new)
print('output3.shape', output3.shape)

plt.imshow(output3[0,:,:], cmap ='gray')
print(f1)
plt.show()

#Prueba 2
f1 = Filter(kernels, bias = 0, padding = 1)
print(f"Filter Bias: {f1.bias}")
output3 = f1.convolve(image_new)
print('output3.shape', output3.shape)

plt.imshow(output3[0,:,:], cmap ='gray')
plt.show()

#Prueba 3: Bias aleatorio
f1 = Filter(kernels, padding = 1)
print(f"Filter Bias: {f1.bias}")
output3 = f1.convolve(image_new)
print('output3.shape', output3.shape)

plt.imshow(output3[0,:,:], cmap ='gray')
plt.show()

#Capas de convolución
l1 = Convolution(2, (3, 3, 3), padding = 1, activation_function = relu)
output4 = l1.forward(image_new)
print('output4.shape', output4.shape)
print(l1)

fig, (ax1, ax2) = plt.subplots(1,2, figsize= (14, 5))

ax1.imshow(output4[0,:,:,0], cmap ='gray')
ax1.set_title('Filtro 1')
ax2.imshow(output4[0,:,:,1], cmap ='gray')
ax2.set_title('Filtro 2')
plt.show()
#Comprobando la disimilitud de la salida de los filtros
output4[0,:,:,0] == output4[0,:,:,1]

#Flatten
output5 = Flatten().forward(output4)
print('output5.shape',output5.shape)