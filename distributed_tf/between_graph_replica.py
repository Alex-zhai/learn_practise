import tensorflow as tf
from multiprocessing import Process
from time import sleep

cluster = tf.train.ClusterSpec({
    "local": ["localhost:2222", "localhost:2223"]
})


def s1():
    server1 = tf.train.Server(cluster, job_name="local", task_index=0)
    sess1 = tf.Session(server1.target)
    print("server 1: running no-op...")
    sess1.run(tf.no_op())
    print("server 1: no-op run!")
    server1.join()


def s2():
    for i in range(3):
        print("server 2: %d seconds left before connecting..." % (3 - i))
        sleep(1.0)
    server2 = tf.train.Server(cluster, job_name="local", task_index=1)
    print("server2: connected!")
    server2.join()

p1 = Process(target=s1, daemon=True)
p2 = Process(target=s2, daemon=True)

p1.start()
p2.start()