import requests
import json
import cv2
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import time

with open('JPY/yen4.JPG', 'rb') as img:
    base64_string = base64.b64encode(img.read()).decode('utf-8')

start_time = time.time()
res = requests.post('http://localhost:5000/predict', data=json.dumps({"img": base64_string}))
print(time.time() -  start_time)

result = json.loads(res.text)
print(result['image'][:30])

encoded_img = np.fromstring(base64.b64decode(result['image']), dtype = np.uint8)
image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

plt.imshow(image)
plt.show()