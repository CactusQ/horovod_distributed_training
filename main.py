import json
import os
import sys
import mnist
import socket
import tensorflow as tf

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure cluster
os.environ["TF_CONFIG"] = '{"cluster": {"worker": ["10.101.0.1", "10.100.0.1"]}, "task": {"type": "worker", "index": 0} }'

# Check if I am the root node (chief node)
my_ip = socket.gethostbyname(socket.gethostname())
if my_ip is not '10.100.0.1':
  os.environ["TF_CONFIG"] = '{"cluster": {"worker": ["10.101.0.1", "10.100.0.1"]}, "task": {"type": "worker", "index": 1} }'

strategy = tf.distribute.MultiWorkerMirroredStrategy()


os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')

batch_size = 64
single_worker_dataset = mnist.mnist_dataset(batch_size)
single_worker_model = mnist.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist.build_and_compile_cnn_model()
