import os
import time
from input_pipeline.HAPT_TFR_dataset import make_HAPT_TFR_Dataset1,make_HAPT_TFR_Dataset2
import tensorflow as tf


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        length = 0
        for sample in dataset:
            # Performing a training step
            # time.sleep(0.001)
            length+=1
    tf.print(length)
    tf.print("Execution time:", time.perf_counter() - start_time)

def main():
    dataset = make_HAPT_TFR_Dataset1('/home/mohamed_alaa/HAPT_TFR/train','load_optimization',250,125,'/home/mohamed_alaa')
    benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE),num_epochs=20)
    # benchmark(dataset,num_epochs=3)





if __name__ == "__main__" :
    main()