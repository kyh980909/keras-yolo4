import requests
import json
import cv2
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import time

with open('JPY/test.png', 'rb') as img:
    base64_string = base64.b64encode(img.read()).decode('utf-8')

start_time = time.time()
res = requests.post('http://0.0.0.0:5000/jpy', data=json.dumps({"img": base64_string})) #58.237.166.159
print(time.time() -  start_time)

result = json.loads(res.text)

encoded_img = np.fromstring(base64.b64decode(result['img']), dtype = np.uint8)
image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

plt.imshow(image)
plt.show()

cv2.imwrite('result.png',image)

total = 0

for key, value in result['count'].items():
    total += int(key[str(key).find('_') + 1:]) * value

# print(result['result'])
print(result['count'])
print(total)