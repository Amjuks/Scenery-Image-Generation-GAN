import io
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input, Dense, Conv2DTranspose, LeakyReLU,
    BatchNormalization, Reshape,
)

def create_generator():
    noise_input = Input(shape=(100,))
    
    x = Dense(16 * 16 * 512, use_bias=False)(noise_input)
    x = Reshape((16, 16, 512))(x)
    
    x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)  
    
    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)  
    
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)  
    
    x = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)  
    
    output = Conv2DTranspose(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')(x)  

    return tf.keras.Model(inputs=noise_input, outputs=output)

def get_layer_outputs(model):
    """
    Create a model that outputs the intermediate outputs of each layer.
    
    Parameters:
    model (tf.keras.Model): The Keras model to extract layer outputs from.
    
    Returns:
    dict: A dictionary where keys are layer names and values are the corresponding layer output models.
    """
    layer_outputs = {}
    for layer in model.layers:
        if hasattr(layer, 'output'):
            intermediate_model = Model(inputs=model.input, outputs=layer.output)
            layer_outputs[layer.name] = intermediate_model
    return layer_outputs

def get_intermediate_outputs(layer_outputs, input_data):
    """
    Get outputs for each layer given input data.
    
    Parameters:
    layer_outputs (dict): Dictionary of layer names to models that output intermediate layers' outputs.
    input_data (np.ndarray): Input data to pass through the models.
    
    Returns:
    dict: A dictionary where keys are layer names and values are the corresponding outputs.
    """
    intermediate_results = {}
    for layer_name, model in layer_outputs.items():
        intermediate_results[layer_name] = model.predict(input_data)
    return intermediate_results

def get_layer_name(layer):    
    if layer == "input_layer":
        return "Random Noise"

    return ' '.join(name for name in layer.split('_') if not name.isdigit()).title()

def highest_factors(x):
    
    highest_pair = (1, x)  

    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            j = x // i
            if i != x and j != x:
                if (i > 1 and j > 1) and (i, j) > highest_pair:
                    highest_pair = (i, j)
    
    
    return highest_pair if highest_pair != (1, x) else None
            
def visualize_intermediate_outputs(outputs, num_cols=4):
    """
    Visualize the intermediate outputs of each layer.

    Parameters:
    outputs (dict): A dictionary where keys are layer names and values are the corresponding outputs.
    num_cols (int): Number of columns in the visualization grid.
    """
    num_layers = len(outputs)
    num_rows = (num_layers + num_cols - 1) // num_cols
    
    figsize = 10
    plt.figure(figsize=(figsize, figsize))
    
    for i, (layer_name, output) in enumerate(outputs.items()):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(get_layer_name(layer_name))
        
        if output.ndim == 4:  
            
            img = output[0]
            
            if img.shape[-1] > 3:
                plt.title(get_layer_name(layer_name) + f'\n{img.shape[-1]} channels')
                img = img[:, :, 0]
            
            img = (img + 1) / 2
            plt.imshow(img)
        
        else:

            if output.shape[1] == 10:
                output = output.reshape(1, -1)
                plt.xticks(np.arange(0, len(output[0]) + 1))
                plt.yticks([])
                plt.imshow(output)
                continue

            else:
                side_x, side_y = highest_factors(output.shape[1])
                output = output.reshape((side_x, side_y))

            plt.imshow(output)
        
        plt.axis('off')
    
    plt.subplots_adjust(wspace=0.04, hspace=0.35)
    buffer = io.BytesIO()
    plt.savefig(buffer, dpi=500)
    buffer.seek(0)
    
    return buffer