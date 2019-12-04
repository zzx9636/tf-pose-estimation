import tensorflow as tf

meta_path = 'models/train3/test/model_latest-4000.meta' # Your .meta file
output_node_names = ['RNN/concat_rnn']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path, clear_devices=True)

    # Load weights
    saver.restore(sess,'models/train3/test/model_latest-4000')
    g = tf.get_default_graph()
    # for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
    #     print(ts)
    # # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # # # Save the frozen graph
    with open('/home/extra_disk/Git_Repo/tf-pose-estimation/models/graph/rnn/graph_opt3.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())