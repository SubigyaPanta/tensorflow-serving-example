import requests
import json
from keras.datasets import mnist
from keras import backend as K


# input image dimensions
from keras.preprocessing.image import img_to_array

img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    print(x_test.shape)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# image = img_to_array(load_img('./data/train/train_10001.jpg', target_size=(128,128))) / 255.
payload = {
  "instances": [{'input_image': img_to_array(x_test[0]).tolist()}]
  # "instances": [{'input_image': [ img_to_array(x_test[i]).tolist() for i in range(5) ]}]
}
r = requests.post('http://localhost:8501/v1/models/viz_mnist:predict',
                  json=payload) # host and port should be the same as in serve.sh
# print(r.content)
response = json.loads(r.content)
print(response)
