
from PIL import Image
from io import BytesIO
import requests
import matplotlib.pyplot as plt
import json
import torchvision

test_set = torchvision.datasets.MNIST(
    root="./",
    train=False,
)

image = test_set.data.numpy()[0]
label = test_set.targets.data.numpy()[0]

image = Image.fromarray(image)
image2bytes = BytesIO()
image.save(image2bytes, format="PNG")
image2bytes.seek(0)
image_as_bytes = image2bytes.read()

response = requests.post("http://localhost:8080/predictions/img_classifier", data=image_as_bytes)
prediction = json.loads(response.text)['prediction']

plt.figure()
plt.title('Label: {}, Prediction: {}'.format(label, prediction))
plt.imshow(image, cmap='gray')
plt.show()

