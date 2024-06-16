import tensorflow as tf
import timeit


# 指定在cpu上运行
def cpu_run():
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([10000, 1000])
        cpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(cpu_a, cpu_b)
    return c


# 指定在gpu上运行
def gpu_run():
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([10000, 1000])
        gpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(gpu_a, gpu_b)
    return c


cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print("cpu:", cpu_time, "  gpu:", gpu_time)