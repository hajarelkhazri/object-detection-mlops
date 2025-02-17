from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import ops
import matplotlib.pyplot as plt
import cv2
import scipy.io
import shutil
import matplotlib.patches as patches
from keras.saving import register_keras_serializable

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(patches, (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels))
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config

def create_vit_object_detector(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units):
    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)
    bounding_box = layers.Dense(4)(features)
    return keras.Model(inputs=inputs, outputs=bounding_box)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        if 'imagefile' not in request.files:
            return "Aucun fichier téléchargé", 400
        imagefile = request.files['imagefile']
        if imagefile.filename == '':
            return "Aucun fichier sélectionné", 400

        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)

        image = keras.utils.load_img(image_path)
        image = image.resize((224, 224))
        image_array = keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        model = keras.models.load_model("vit_object_detector.keras")
        preds = model.predict(image_array)[0]

        (h, w) = image.size
        top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)
        bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')

        predicted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_' + filename)
        plt.savefig(predicted_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return render_template('index.html', filename='predicted_' + filename)

    except Exception as e:
        return f"Une erreur s'est produite : {str(e)}", 500

@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)