import matplotlib.pyplot as plt
import requests
import json
import numpy as np
from keras.datasets import mnist

(_, _), (x_test, y_test) = mnist.load_data()
num=6
images_to_predict=x_test[:num]
label_to_predict=y_test[:num]
x_test = x_test.astype('float32')
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

url = 'http://localhost:8501/v1/models/img_classifier:predict'
data = json.dumps({"instances": x_test[:num].tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post(url, data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

num_row = 2
num_col = 3
fig, axes = plt.subplots(num_row, num_col, figsize=(1*0.8*num_col,1.5*0.8*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images_to_predict[i], cmap='gray')
    ax.set_title('Label: {}, Prediction: {}'.format(label_to_predict[i], np.argmax(predictions[i])))

plt.tight_layout()
plt.show()
