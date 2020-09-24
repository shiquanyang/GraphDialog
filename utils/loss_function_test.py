import torch.nn as nn
import numpy as np
import torch
import tensorflow as tf
import pdb


def gen_series(cnt):
    i = 0
    while True:
        size = np.random.randint(2, 10)
        rand_list = np.random.normal(size=(size,))
        elm = [[100, 200, 300]]
        elm2 = [1000, 2000, 3000]
        for j in rand_list:
            elm2 = elm2 + elm2 * (j+1).astype(np.int)
            for k in range(j.astype(np.int)):
                elm.append([100, 200, 300])
        yield i, elm, elm2
        i += 1
# j = 0
# for i in gen_series():
#     if j < 10:
#         print(i)
#         j += 1
#     else:
#         break

ds_series = tf.data.Dataset.from_generator(lambda: gen_series(1), output_types=(tf.int32, tf.int32, tf.int32), output_shapes=((), (None, None), (None,)))
print(ds_series)
# pdb.set_trace()
# t = ds_series.shuffle(20)
ds_series_batch = ds_series.shuffle(20).padded_batch(10, padded_shapes=([], [None, 3], [None]), padding_values=(0, 1, 1))
ids, context_arr, response = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(context_arr.numpy())
print()
print(response.numpy())


# sigmoid = nn.Sigmoid()
# out = sigmoid(torch.Tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]))
#
# loss = nn.BCELoss()
# torch_bce_loss = loss(torch.Tensor(sigmoid(torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))), torch.Tensor([[1, 0, 1], [1, 1, 1]]))
# print("torch_bce_loss:", torch_bce_loss)
#
# tensorflow_bce_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.cast(tf.convert_to_tensor([[1, 0, 1], [1, 1, 1]]), dtype=tf.double), tf.cast(tf.convert_to_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), dtype=tf.double))
# tensorflow_bce_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(tf.cast(tf.convert_to_tensor([[1, 0, 1], [1, 1, 1]]), dtype=tf.double), tf.cast(tf.convert_to_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), dtype=tf.double))
# loss2 = tf.cast(tf.reduce_sum(tensorflow_bce_loss2)/(tensorflow_bce_loss2.shape[0]*tensorflow_bce_loss2.shape[1]), dtype=tf.float32)
# print("tensorflow_bce_loss:", tensorflow_bce_loss)
# print("tensorflow_bce_loss2:", loss2)
