from tqdm import tqdm
from utils.config import *
from tensorflow_models.GLMPGraph_with_gate import *
import datetime
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

early_stop = args['earlyStop']
if args['dataset'] == 'kvr':
    from utils.utils_tensorflow_Ent_gate_learning import *
    # from utils.utils_tensorflow_Ent_graphgru import *
    early_stop = 'ENTF1'
elif args['dataset'] == 'babi':
    from utils.utils_Ent_babi import *
    early_stop = None
    if args['task'] not in ['1', '2', '3', '4', '5']:
        print("[ERROR] You need to provide the correct --task information.")
        exit(1)
else:
    print("[Error] You need to provide the dataset information.")

print("Is there a GPU available: ", tf.test.is_gpu_available())


# ===============================
# Configure models and load data
# ===============================
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, testOOV, lang, max_response_len, train_length, dev_length, test_length = prepare_data_seq(args['task'],
                                                                      batch_size=int(args['batch']))

# ===============================
# Build model
# ===============================
model = GLMPGraph(int(args['hidden']),
                  lang,
                  max_response_len,
                  args['path'],
                  args['task'],
                  lr=float(args['learn']),
                  n_layers=int(args['layer']),
                  graph_hidden_size=int(args['graphhdd']),
                  nheads=int(args['nheads']),
                  alpha=float(args['alpha']),
                  dropout=float(args['drop']),
                  graph_dr=float(args['graph_dr']),
                  n_graph_layers=int(args['graph_layer']))

# ================================
# Training
# ================================
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)

for epoch in range(200):
    print("Epoch:{}".format(epoch))
    # pdb.set_trace()
    train_length_ = compute_dataset_length(train_length, int(args['batch']))
    pbar = tqdm(enumerate(train.take(-1)), total=(train_length_))
    for i, data in pbar:
        tf.config.experimental_run_functions_eagerly(True)
        model.train_batch(data, int(args['clip']), reset=(i==0))
        tf.config.experimental_run_functions_eagerly(False)
        pbar.set_description(model.print_loss())

    # with train_summary_writer.as_default():
    #     tf.summary.scalar('loss', model.train_loss.result(), step=epoch)
    #     tf.summary.scalar('loss_g', model.train_loss_g.result(), step=epoch)
    #     tf.summary.scalar('loss_v', model.train_loss_v.result(), step=epoch)
    #     tf.summary.scalar('loss_l', model.train_loss_l.result(), step=epoch)
    # model.train_loss.reset_states()
    # model.train_loss_g.reset_states()
    # model.train_loss_v.reset_states()
    # model.train_loss_l.reset_states()

    if ((epoch+1) % int(args['evalp']) == 0):
        # len = int(dev_length / (int(args['batch'])))
        dev_length_ = compute_dataset_length(dev_length, int(args['batch']))
        acc = model.evaluate(dev, dev_length_, avg_best, early_stop)

        if (acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1

        if (cnt == 8 or (acc == 1.0 and early_stop == None)):
            print("Run out of patient, early stop...")
            break
