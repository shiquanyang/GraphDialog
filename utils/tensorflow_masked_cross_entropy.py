import tensorflow as tf
import pdb


def sequence_mask(sequence_length, max_len=None):
    '''
    Generate mask matrix according to sequence length information.
    :param sequence_length:
    :param max_len:
    :return:
    '''
    if max_len is None:
        max_len = sequence_length.numpy().max()
    batch_size = sequence_length.shape[0]
    seq_range = tf.range(0, max_len)  # 1 * max_len.
    seq_range_expand = tf.tile(tf.expand_dims(seq_range, 0), [batch_size, 1])  # seq_range_expand: batch_size * max_len.
    seq_length_expand = tf.tile(tf.expand_dims(sequence_length, 1), [1, seq_range_expand.shape[1]])
    return tf.cast((seq_range_expand < seq_length_expand), dtype=tf.int32)


def generate_indices(target_flat):
    '''
    Generate slice index matrix to provide input for tf.gather_nd.
    :param target:
    :return:
    '''
    max_len = target_flat.shape[0]  # max_len: (batch_size * max_len).
    indices = [[i, target_flat[i, 0]] for i in range(max_len)]  # indices: (batch_size * max_len) * 2.
    return indices


def masked_cross_entropy(logits, target, length):
    '''
    Masked cross entropy loss at timestep granularity.
    The sigmoid_cross_entropy_with_logits in tensorflow
    only support at sample granularity.

    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index(zero-based) of the true class for each corresponding step,
            the value should be in the range of [0, num_classes-1].
        length: A Variable containing a LongTensor of size
            (batch,) which contains the length of each data in a batch,
            should contain at least one non-zero number, each number should be in the range (0, max_len].
    Returns:
        loss: An average loss value for single timestep masked by the length.
    '''
    # pdb.set_trace()
    logits_flat = tf.reshape(logits, [-1, logits.get_shape()[-1]])  # logits: batch_size * max_len * num_classes, logits_flat: (batch_size * max_len) * num_classes.
    log_probs_flat = tf.nn.log_softmax(logits_flat)  # log_probs_flat: (batch_size * max_len) * num_classes.
    target_flat = tf.reshape(target, [-1, 1])  # target_flat: (batch_size * max_len) * 1.
    losses_flat = -tf.gather_nd(log_probs_flat, generate_indices(target_flat))  # loss_flat: (batch_size * max_len) * 1.
    losses = tf.reshape(losses_flat, target.shape)  # losses: batch_size * max_len.
    mask = sequence_mask(sequence_length=length, max_len=target.shape[1])
    # print(losses)
    # print(mask)
    losses = losses * tf.cast(mask, tf.float32)
    # print(losses)
    loss = tf.reduce_sum(losses) / tf.reduce_sum(tf.cast(length, tf.float32))
    return loss


# ==============================
# Unit Test
# ==============================
if __name__ == '__main__':
    ret = masked_cross_entropy(tf.Variable([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]],
                                            [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04], [0.0, 0.0, 0.0]]]), tf.Variable([[0, 2, 1], [1, 0, 0]]), tf.Variable([2, 2]))
    print(ret)
