import sys
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

print('load onnx', end='->')
onnx_model_path = sys.argv[1]
onnx_model = onnx.load(onnx_model_path)

tf_rep = prepare(onnx_model)
print('prepare tf', end='->')

tf_model_path = onnx_model_path.split('.')[0]

name = tf_model_path
if "defo" in name:
    input_shape = [1, 3, 108, 108]
elif name in ['vit_base_16_112', 'vit_base_8_112']:
    input_shape = [1, 3, 112, 112]
elif name in ['vit_tiny_224', 'vit_base_224']:
    input_shape = [1, 3, 224, 224]
elif name =='vit_tiny_16_112':
    input_shape = [1, 3, 112, 112]
elif name =='vit_tiny_8_112':
    input_shape = [1, 3, 112, 112]
elif name =='beit_base':
    input_shape = [1, 3, 224, 224]
elif name =='mobilevit_108':
    input_shape = [1, 3, 108, 108]
elif name =='mobilevit_256':
    input_shape = [1, 3, 256, 256]

print('export graph', end='->')
tf_rep.export_graph(tf_model_path)

print('load pb', end='->')
model = tf.saved_model.load(tf_model_path)
model.trainable = False

input_tensor = tf.random.uniform(input_shape)
out = model(**{'input0': input_tensor})
print('convert the model', end='->')
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

tflite_model_path = tf_model_path + '.tflite'
print(' Save the model')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
