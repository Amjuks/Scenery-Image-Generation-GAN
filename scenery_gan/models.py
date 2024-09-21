import io
import os
import re

import moviepy.editor as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from math import ceil, sqrt
from pathlib import Path

from PIL import Image
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Conv2DTranspose, LeakyReLU,
    BatchNormalization, Reshape, Flatten, Dropout
)

from .visualize import get_layer_outputs, get_intermediate_outputs, visualize_intermediate_outputs

CHECKPOINT_FILE_PATTERN = re.compile(r'ckpt-(\d+)\.weights\.h5')
PROGRESS_IMAGES_FILE_PATTERN = re.compile(r'img_epoch_(\d+)\.(png|jpg)')

BASE_DIR = Path(__file__).parent

GENERATOR_CKPT_DIR = BASE_DIR / "generator_checkpoints"
EPOCH_IMAGES_DIR = BASE_DIR / "epoch_images"
PROGRESS_GIFS_DIR = BASE_DIR / 'progress_gifs'

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

def create_discriminator():
    image_input = Input(shape=(256, 256, 3))
    
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(image_input)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512)(x)  
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)  

    output = Dense(1, activation='sigmoid')(x)  

    return tf.keras.Model(inputs=image_input, outputs=output)

def get_dir_files(directory: str, pattern: re.Pattern[str]) -> list[str]:
    dir_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)

            if match:
                index = int(match.group(1))
                dir_files.append((index, os.path.join(root, file)))

    dir_files.sort(reverse=False, key=lambda x: x[0])
    return [filename for _, filename in dir_files]

class SceneryGAN:
    def __init__(
            self,
            generator_checkpoints_dir: str = GENERATOR_CKPT_DIR,
            progress_images_dir: str = EPOCH_IMAGES_DIR,
            *,
            visualize: bool = False,
            checkpoint: int | str = -1,
            resolution: int = 100,
    ) -> None:

        if not isinstance(resolution, int):
            raise ValueError(f"Expected int for `resolution` recieved {type(resolution)}")
        self.resolution = resolution

        if not os.path.isdir(generator_checkpoints_dir):
            raise ValueError(f"The path '{generator_checkpoints_dir}' is not a valid directory.")

        if not os.path.isdir(progress_images_dir):
            raise ValueError(f"The path '{progress_images_dir}' is not a valid directory.")
        
        self.generator_checkpoints = get_dir_files(generator_checkpoints_dir, CHECKPOINT_FILE_PATTERN)

        self.generator_checkpoint = None

        self.progress_images = get_dir_files(progress_images_dir, pattern=PROGRESS_IMAGES_FILE_PATTERN)

        self.generator_model = create_generator()
    
        if self.generator_checkpoints:
            self.load_generator_weights(checkpoint)
            
        if visualize:
            self.set_visualizer()

    def set_visualizer(self):
        self.intermediate_layer_outputs = get_layer_outputs(self.generator_model)

    def load_generator_weights(self, weights_file: str | int):

        if isinstance(weights_file, int):
            self.generator_model.load_weights(self.generator_checkpoints[int(weights_file)])
        elif isinstance(weights_file, str):   
            self.generator_model.load_weights(weights_file)
        else:
            raise ValueError(f"Expected str or int, recieved {type(weights_file)}")

        self.generator_checkpoint = weights_file
        self.set_visualizer()

    def reset_generator(self):
        self.load_generator_weights(0)

    def random_noise(self, num_samples=1, noise_dim=100):
        return tf.random.normal((num_samples, noise_dim))

    def create_images(self, images, *, real_image=False):

        num_images = len(images)
        grid_size = ceil(sqrt(num_images))

        plt.figure(figsize=(grid_size*3, grid_size*3))

        if real_image:
            for i in range(num_images):
                image = plt.imread(images[i])
                plt.subplot(grid_size, grid_size, i+1)
                plt.imshow(image)
                plt.axis('off')

        else:
            for i in range(num_images):
                image = ((images[i] + 1) / 2)
                plt.subplot(grid_size, grid_size, i+1)
                plt.imshow(image)
                plt.axis('off')

        buffer = io.BytesIO()
        plt.savefig(buffer, dpi=self.resolution)
        buffer.seek(0)

        return buffer
            
    def show_images(self, images):
        Image.open(images).show()

    def generate_images(self, num_images: int = 9, *, show=False, noise=None):
        if noise is None:
            noise = self.random_noise(num_images)

        images = self.generator_model.predict(noise)

        if show:
            self.show_images(self.create_images(images))

        return images
    
    def generate_overtime_images(self, *, show=False):
        initial_checkpoint = self.generator_checkpoint
        images = []
        num_checkpoints = len(self.generator_checkpoints)
        noise = self.random_noise()

        for i in range(0, num_checkpoints-1):
            self.load_generator_weights(i)
            images.append(self.generate_images(1, noise=noise)[0])

        self.load_generator_weights(initial_checkpoint)

        if show:
            self.show_images(self.create_images(images))
        
        return images
    
    def generate_multiple_overtime_images(self, *, show=False):
        initial_checkpoint = self.generator_checkpoint
        images = []
        num_checkpoints = len(self.generator_checkpoints)
        num_images = 4
        noise = self.random_noise(num_images)

        for i in range(num_checkpoints - 1, -1, -1):
            self.load_generator_weights(i)
            image = self.generate_images(num_images, noise=noise)
            images.append(self.create_images(image))

        self.load_generator_weights(initial_checkpoint)
        
        if show:
            self.show_images(self.create_images(images, real_image=True))
        
        return images
        
    def generate_visualizer_image(self, num_images=1,  *, show=False, noise=None):
        images = []
        if noise is None:
            noise = self.random_noise()

        for i in range(num_images):
            generator_intermediate_outputs = get_intermediate_outputs(self.intermediate_layer_outputs, noise)
            images.append(visualize_intermediate_outputs(generator_intermediate_outputs))

        if show:
            self.show_images(self.create_images(images, real_image=True))

        return images
    
    def progress_gif(
        self,
        speed: int = 500,
        name: str = 'output',
        directory: str = PROGRESS_GIFS_DIR,
        *,
        loop: int = 0,
        video: bool = False
    ):
        images = [Image.open(img) for img in self.progress_images]

        filename = os.path.join(directory, name)
        images[0].save(
            filename + '.gif',
            format='GIF',
            save_all=True,
            append_images=images[1:],
            duration=speed,
            loop=loop
        )

        if video:
            clip = mp.VideoFileClip(filename + '.gif')
            clip.write_videofile(filename + '.mp4')