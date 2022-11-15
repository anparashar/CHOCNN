import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import time
import keras
import numpy as np
import keras.backend
import matplotlib
import matplotlib.pyplot as plt
from psoCNN import psoCNN
#import tensorflow.compat.v1 as tf
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


if __name__ == '__main__':
    ######## Algorithm parameters ##################
    
    dataset = "cifar10"
    # dataset = "mnist"
    # dataset = "mnist-rotated-digits"
    # dataset = "mnist-rotated-with-background"
    # dataset = "rectangles"
    # dataset = "rectangles-images"
    # dataset = "convex"
    # dataset = "fashion-mnist"
    # dataset = "mnist-random-background"
    # dataset = "mnist-background-images"
    
    number_runs = 1
    number_iterations = 10
    population_size = 20

    batch_size_pso = 128
    batch_size_full_training = 128
    
    epochs_pso = 10
    epochs_full_training = 100
    
    max_conv_output_channels = 256
    max_fully_connected_neurons = 128

    max_conv_kernel_size = 7
    max_pool_kernel_size = 7

    dropout = 0.25

    ########### Run the algorithm ######################
    results_path = "./results/" + dataset + "/"

    if not os.path.exists(results_path):
            os.makedirs(results_path)

    all_gBest_metrics = np.zeros((number_runs, 2))
    runs_time = []
    all_gbest_par = []
    all_gbest_err = []
    best_gBest_acc = 0

    for i in range(number_runs):
        print("Run number: " + str(i))
        start_time = time.time()
        pso = psoCNN(dataset=dataset, n_iter=number_iterations, pop_size=population_size, \
            batch_size=batch_size_pso, epochs=epochs_pso, max_conv_kernel=max_conv_kernel_size, \
            max_out_ch=max_conv_output_channels, max_pool_kernel=max_pool_kernel_size, max_fc_neurons=max_fully_connected_neurons, dropout_rate=dropout, results_path= results_path)
 
        pso.fit(dropout_rate=dropout)

        print(pso.gBest_acc)

        # Plot current gBest
        matplotlib.use('Agg')
        plt.plot(pso.gBest_acc)
        plt.xlabel("Iteration")
        plt.ylabel("gBest acc")
        plt.savefig(results_path + "gBest-iter-" + str(i) + ".png")
        plt.close()

        end_time = time.time()

        running_time = end_time - start_time

        runs_time.append(running_time)

        output_str = "This run took: " + str(running_time) + " seconds.\n\n\n" + 'gBest_pso_metric(iter, accuracy, loss_test)\n' + str(pso.gBest_pso_metric) + "\n\n" +'gbest index :' + str(pso.gBest_index)+ '\n'
        all_gBest = np.zeros((number_iterations, 1))
        for i in range(len(pso.gBest_i)):
          print('gBest [' + str(i) + '] : ' + str(pso.gBest_i[i]))
          print('Vel gBest [' + str(i) + '] : ' + str(pso.gBest_i[i].vel) + '\n')

        print(output_str)

        with open(results_path + "/final_results.txt", "w") as f:
            try:
                print(output_str, file=f)
            except SyntaxError:
                print >> f, output_str