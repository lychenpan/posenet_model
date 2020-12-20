import tensorflow as tf
from PIL import Image
import numpy as np
import time

path = "/home/cp/Develop/Workspace/pose-estimation/pose-models/model.tflite"

interpreter = tf.lite.Interpreter(path)

input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

t1 = "/home/cp/Desktop/test1.png"

timg = Image.open(t1)

timg2 = timg.resize((192,192))

tim = np.array(timg2)

tim2 = np.expand_dims(tim, axis=0)

t1 = time.time()
interpreter.set_tensor(input_details[0]['index'], tim2.astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
t2 = time.time()

print(t2-t1)
print(output_data)
