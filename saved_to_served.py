from keras.engine.saving import load_model
from tensorflow.contrib.saved_model import save_keras_model
from keras import backend as K
import tensorflow as tf


version = '001'
serve_model_dir = f'./serve_models/viz_mnist/{version}'
model_dir = './models/'
model_name = 'third'
model = load_model(f'{model_dir}{model_name}.hdf5')

with K.get_session() as sess:
    tf.saved_model.simple_save(sess,
                               serve_model_dir,
                               inputs={'input_image': model.input},
                               outputs={t.name:t for t in model.outputs})
# save_keras_model(model, serve_model_dir)

