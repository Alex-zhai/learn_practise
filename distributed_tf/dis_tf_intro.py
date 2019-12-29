import tensorflow as tf

# var = tf.Variable(initial_value=0.0)
#
# sess1 = tf.Session()
# sess2 = tf.Session()
#
# sess1.run(tf.global_variables_initializer())
# sess2.run(tf.global_variables_initializer())
#
# print("Initial value of var in session 1:", sess1.run(var))
# print("Initial value of var in session 2:", sess2.run(var))
#
# sess1.run(var.assign_add(1.0))
# print("Increment var in session 1")
#
# print("Value of var in session 1:", sess1.run(var))
# print("Value of var in session 2:", sess2.run(var))
#
# tasks = ["localhost:2222", "localhost:2223"]
# jobs = {"local": tasks}
#
# # cluster = tf.train.ClusterSpec(
# #     {"worker": ["worker0.example.com:2222", "worker1.example.com:2222", "worker2.example.com:2222"],
# #      "ps": ["ps0.example.com:2222", "ps1.example.com:2222"]})
# cluster = tf.train.ClusterSpec(jobs)
# server1 = tf.train.Server(cluster, job_name="local", task_index=0)
# server2 = tf.train.Server(cluster, job_name="local", task_index=1)
#
# tf.reset_default_graph()
# var = tf.Variable(initial_value=0.0, name='var')
# sess1 = tf.Session(server1.target)
# sess2 = tf.Session(server2.target)
#
# sess1.run(tf.global_variables_initializer())
# sess2.run(tf.global_variables_initializer())
#
# print("Initial value of var in session 1:", sess1.run(var))
# print("Initial value of var in session 2:", sess2.run(var))
#
# sess1.run(var.assign_add(1.0))
# print("Incremented var in session 1")
#
# print("Value of var in session 1:", sess1.run(var))
# print("Value of var in session 2:", sess2.run(var))
#
#
# def run_with_location_trace(sess, op):
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     sess.run(op, options=run_options, run_metadata=run_metadata)
#     for device in run_metadata.step_stats.dev_stats:
#         print(device.device)
#         for node in device.node_stats:
#             print("  ", node.node_name)
#
# run_with_location_trace(sess1, var)
# run_with_location_trace(sess1, var.assign_add(1.0))
# run_with_location_trace(sess2, var)
#
# with tf.device("/job:local/task:0"):
#     var1 = tf.Variable(0.0, name='var1')
#
# with tf.device("/job:local/task:1"):
#     var2 = tf.Variable(0.0, name='var2')
#
# sess1.run(tf.global_variables_initializer())
#
# run_with_location_trace(sess1, var1)
#
# run_with_location_trace(sess1, var2)

cluster = tf.train.ClusterSpec({"local": ["localhost: 2224", "localhost: 2225"]})
server1 = tf.train.Server(cluster, job_name="local", task_index=0)
server2 = tf.train.Server(cluster, job_name="local", task_index=1)

graph1 = tf.Graph()
with graph1.as_default():
    var1 = tf.Variable(0.0, name='var')
sess1 = tf.Session(target=server1.target, graph=graph1)
print(graph1.get_operations())

graph2 = tf.Graph()
sess2 = tf.Session(target=server2.target, graph=graph2)
print(graph2.get_operations())

with graph2.as_default():
    var2 = tf.Variable(0.0, name='var')
sess1.run(var1.assign(1.0))
sess2.run(var2)


