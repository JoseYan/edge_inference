import numpy as np
import tensorflow as tf
import sys
from time import time
from tqdm import tqdm

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
inputs = [np.array(np.random.random_sample(input_shape), dtype=np.float32) for _ in range(100)]

t0 = time()
for inp in tqdm(inputs):
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
t1 = time()

# get_tensor() returns a copy of the tensor data
# use tensor() in order to get a pointer to the tensor
mname= sys.argv[1].split('.')[0]

print(f'{mname}:{(t1-t0)*10:.4f}')
with open('tflite_inf.log','a') as f:
    f.write(f'{mname}:{(t1-t0)*10:.4f}\n')

