import tensorflow as tf
from utils.config import *


def gen_samples(data, length):
    def compute_length(sequences):
        lengths = [len(seq) for seq in sequences]
        return lengths

    # change information format to category-wise
    data_info = {}
    for key in data[0].keys():
        data_info[key] = [d[key] for d in data]

    # compute additional information, e.g, lengths
    context_arr_lengths = compute_length(data_info['context_arr'])
    response_lengths = compute_length(data_info['response'])
    selector_lengths = compute_length(data_info['selector_index'])
    ptr_lengths = compute_length(data_info['ptr_index'])
    conv_arr_lengths = compute_length(data_info['conv_arr'])
    sketch_response_lengths = compute_length(data_info['sketch_response'])
    kb_arr_lengths = compute_length(data_info['kb_arr'])
    context_arr_plain_lengths = compute_length(data_info['context_arr_plain'])
    kb_arr_plain_lengths = compute_length(data_info['kb_arr_plain'])
    ent_index_lengths = compute_length(data_info['ent_index'])
    ent_idx_cal_lengths = compute_length(data_info['ent_idx_cal'])
    ent_idx_nav_lengths = compute_length(data_info['ent_idx_nav'])
    ent_idx_wet_lengths = compute_length(data_info['ent_idx_wet'])

    # add additional information
    data_info['context_arr_lengths'] = context_arr_lengths
    data_info['response_lengths'] = response_lengths
    data_info['conv_arr_lengths'] = conv_arr_lengths
    data_info['kb_arr_lengths'] = kb_arr_lengths
    data_info['ent_index_lengths'] = ent_index_lengths
    data_info['ent_idx_cal_lengths'] = ent_idx_cal_lengths
    data_info['ent_idx_nav_lengths'] = ent_idx_nav_lengths
    data_info['ent_idx_wet_lengths'] = ent_idx_wet_lengths

    # yield output tuple
    for i in range(length):
        yield data_info['context_arr'][i],\
              data_info['response'][i],\
              data_info['sketch_response'][i],\
              data_info['conv_arr'][i],\
              data_info['ptr_index'][i],\
              data_info['selector_index'][i],\
              data_info['kb_arr'][i],\
              data_info['context_arr_plain'][i],\
              data_info['response_plain'][i],\
              data_info['kb_arr_plain'][i],\
              data_info['context_arr_lengths'][i],\
              data_info['response_lengths'][i],\
              data_info['conv_arr_lengths'][i],\
              data_info['kb_arr_lengths'][i],\
              data_info['ent_index'][i],\
              data_info['ent_index_lengths'][i],\
              data_info['ent_idx_cal'][i],\
              data_info['ent_idx_nav'][i],\
              data_info['ent_idx_wet'][i],\
              data_info['ent_idx_cal_lengths'][i],\
              data_info['ent_idx_nav_lengths'][i],\
              data_info['ent_idx_wet_lengths'][i],\
              data_info['ID'][i],\
              data_info['deps'][i],\
              data_info['deps_type'][i],\
              data_info['cell_masks'][i]


def get_seq(data_info, batch_size, drop_remainder):
    ds_series = tf.data.Dataset.from_generator(lambda: gen_samples(data_info, len(data_info)),
                                   output_types=(tf.int32,  # context_arr 0
                                                 tf.int32,  # response 1
                                                 tf.int32,  # sketch_response 2
                                                 tf.int32,  # conv_arr 3
                                                 tf.int32,  # ptr_index 4
                                                 tf.int32,  # selector_index 5
                                                 tf.int32,  # kb_arr 6
                                                 tf.string,  # context_arr_plain 7
                                                 tf.string,  # response_plain 8
                                                 tf.string,  # kb_arr_plain 9
                                                 tf.int32,  # context_arr_lengths 10
                                                 tf.int32,  # response_lengths 11
                                                 tf.int32,  # conv_arr_lengths 12
                                                 tf.int32,  # kb_arr_lengths 13
                                                 tf.string,  # ent_index 14
                                                 tf.int32,  # ent_index_lengths 15
                                                 tf.string,  # ent_idx_cal 16
                                                 tf.string,  # ent_idx_nav 17
                                                 tf.string,  # ent_idx_wet 18
                                                 tf.int32,  # ent_idx_cal_lengths 19
                                                 tf.int32,  # ent_idx_nav_lengths 20
                                                 tf.int32,  # ent_idx_wet_lengths 21
                                                 tf.int32,  # ID 22
                                                 tf.string,  # deps 23
                                                 tf.int32,  # deps_type 24
                                                 tf.int32  # cell_masks 25
                                                 ),
                                   output_shapes=((None, None),  # context_arr
                                                  (None,),  # response
                                                  (None,),  # sketch_response
                                                  (None, None),  # conv_arr
                                                  (None,),  # ptr_index
                                                  (None,),  # selector_index
                                                  (None, None),  # kb_arr
                                                  (None, None),  # context_arr_plain
                                                  (None,),  # response_plain
                                                  (None, None),  # kb_arr_plain
                                                  (),  # context_arr_lengths
                                                  (),  # response_lengths
                                                  (),  # conv_arr_lengths
                                                  (),  # kb_arr_lengths
                                                  (None,),  # ent_index
                                                  (),  # ent_index_lengths
                                                  (None,),  # ent_idx_cal
                                                  (None,),  # ent_idx_nav
                                                  (None,),  # ent_idx_wet
                                                  (),  # ent_idx_cal_lengths
                                                  (),  # ent_idx_nav_lengths
                                                  (),  # ent_idx_wet_lengths
                                                  (),  # ID
                                                  (None, None, None),  # deps
                                                  (None, None, None),  # deps_type
                                                  (None, None, None)  # cell_masks
                                                  )
                                   )
    print(len(data_info))
    ds_series_batch = ds_series.shuffle(len(data_info)).padded_batch(batch_size, padded_shapes=([None, MEM_TOKEN_SIZE],  # context_arr
                                                                              [None,],  # response
                                                                              [None,],  # sketch_response
                                                                              [None, MEM_TOKEN_SIZE],  # conv_arr
                                                                              [None,],  # ptr_index
                                                                              [None,],  # selector_index
                                                                              [None, MEM_TOKEN_SIZE],  # kb_arr
                                                                              [None, MEM_TOKEN_SIZE],  # context_arr_plain
                                                                              [None,],  # response_plain
                                                                              [None, MEM_TOKEN_SIZE],  # kb_arr_plain
                                                                              [],  # context_arr_lengths
                                                                              [],  # response_lengths
                                                                              [],  # conv_arr_lengths
                                                                              [],  # kb_arr_lengths
                                                                              [None,],  # ent_index
                                                                              [],  # ent_index_lengths
                                                                              [None,],  # ent_idx_cal
                                                                              [None,],  # ent_idx_nav
                                                                              [None,],  # ent_idx_wet
                                                                              [],  # ent_idx_cal_lengths
                                                                              [],  # ent_idx_nav_lengths
                                                                              [],  # ent_idx_wet_lengths
                                                                              [],  # ID
                                                                              [2, None, MAX_DEPENDENCIES_PER_NODE],  # deps
                                                                              [2, None, (MAX_DEPENDENCIES_PER_NODE+1)],  # deps_type
                                                                              [2, None, (MAX_DEPENDENCIES_PER_NODE+1)]  # cell_masks
                                                                              ),
                                                   padding_values=(PAD_token,  # context_arr
                                                                   PAD_token,  # response
                                                                   PAD_token,  # sketch_response
                                                                   PAD_token,  # conv_arr
                                                                   PAD_token,  # ptr_index
                                                                   0,  # selector_index
                                                                   PAD_token,  # kb_arr
                                                                   'PAD',  # context_arr_plain
                                                                   'PAD',  # response_plain
                                                                   'PAD',   # kb_arr_plain
                                                                   0,  # context_arr_lengths
                                                                   0,  # response_lengths
                                                                   0,  # conv_arr_lengths
                                                                   0,  # kb_arr_lengths
                                                                   'PAD',  # ent_index
                                                                   0,  # ent_index_lengths
                                                                   'PAD',  # ent_idx_cal
                                                                   'PAD',  # ent_idx_nav
                                                                   'PAD',  # ent_idx_wet
                                                                   0,  # ent_idx_cal_lengths
                                                                   0,  # ent_idx_nav_lengths
                                                                   0,  # ent_idx_wet_lengths
                                                                   0,  # ID
                                                                   '$',  # deps
                                                                   PAD_token,  # deps_type
                                                                   0  # cell_masks
                                                                   ),
                                                    drop_remainder=drop_remainder
                                                   )
    ds_series_batch = ds_series_batch.prefetch(1)
    return ds_series_batch
