import os
import sys
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow._api.v2 import data
from tensorflow.python.types.core import Value

# DEFAULT TRAINING SETTINGS
total_training_batches = 100
batch_size = 64

try:
    if len(sys.argv) > 1:
        total_training_batches = int(sys.argv[1])
    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
except ValueError as e:
    print ("Value error.")
    print (f"Usage: python {sys.argv[0]} <total_num_batches> <batch_size>")

# Initialize Horovod
hvd.init()

# Download entire MNIST dataset
(mnist_images, mnist_labels), _  = \
    tf.keras.datasets.mnist.load_data(path='mnist.npz')

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)

training_slice_per_worker = len(mnist_images) // hvd.size()

# Print training metadata
print(f"===========")
print(f"rank: {hvd.rank()}")
print(f"local_rank: {hvd.local_rank()}")
if hvd.rank() == 0:
    print(f"num of workers: {hvd.size()}")
    print(f"Slice per worker: {training_slice_per_worker}")
    print(f"Batch size: {batch_size}")
    print(f"Total amount of batches: {total_training_batches}")
    print(f"Total amount of examples: {total_training_batches * batch_size}")
print(f"===========")


# Split dataset so that each worker has unique slice of entire MNIST data
dataset = dataset.skip(training_slice_per_worker*hvd.rank())
dataset = dataset.take(training_slice_per_worker)

# Shuffle dataset (pseudo-random, so we can reproduce the same splits for each trial)
dataset = dataset.repeat() \
                .shuffle(total_training_batches, seed=42) \
                .batch(batch_size)

# Build the DNN model (deep neural network)
mnist_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)
])


# Define optimizer
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.optimizers.Adam(0.001 * hvd.size())

mnist_model.compile(
    loss=loss,
    optimizer=opt,
    metrics=['accuracy'])

checkpoint_dir = './checkpoints'
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)

accuracy_metric = tf.keras.metrics.Accuracy()

@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    # Broadcast initial model parameters and optimizer variables (sync training step)
    # This is equivalent to MPI_BCAST
    if first_batch:
        hvd.broadcast_variables(mnist_model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    return loss_value

# Split total number of training batches across all workers
for batch_num, (images, labels) in enumerate(dataset.take(total_training_batches // hvd.size())):
    loss_value = training_step(images, labels, batch_num == 0)

    # Printout progress every 10 steps (1 step = 1 batch processed)
    if batch_num % 10 == 0 and hvd.local_rank() == 0:
        print('Step #%d (total examples = %d)\tLoss: %.6f' % \
            (batch_num, batch_num*batch_size, loss_value))

# Save model at checkpoint direction (will be overridden at each trial)
# The checkpoint will not be automatically loaded, 
# So all trials are independent from each other
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)

    # Evaluate final model with unseen test data from MNIST (10000 examples)
    _, (test_images, test_labels)= \
        tf.keras.datasets.mnist.load_data(path='mnist-2.npz')

    results = mnist_model.evaluate(test_images, test_labels, return_dict=True)
    accuracy = results["accuracy"]
    print('Final model accuracy: %.6f' % (accuracy))