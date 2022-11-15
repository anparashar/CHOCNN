import keras
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
import keras.backend
from population import Population
import numpy as np
from copy import deepcopy
import os
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
class psoCNN:
    def __init__(self, dataset, n_iter, pop_size, batch_size, epochs, max_conv_kernel, max_out_ch, max_pool_kernel, max_fc_neurons, dropout_rate, results_path):
        self.results_path = results_path
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.epochs = epochs
        self.gBest_index = None

        self.batch_size = batch_size
        self.gBest_acc = np.zeros(n_iter)
        self.gBest_test_acc = np.zeros(n_iter)

        self.gBest_i = []
        self.num_gBest = 0
        self.gBest_pso_metric = np.zeros((n_iter, 3))  # iter, acc_test,loss_test
        
        if dataset == "cifar10":
            input_width = 32
            input_height = 32
            input_channels = 3
            output_dim = 10

            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_test.astype('float32')
            mu = np.mean(self.x_train, axis=0)
            self.x_train -= mu
            self.x_train /= 255
            
            self.x_test -= mu
            self.x_test /= 255

        if dataset == "mnist":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        
        if dataset == "fashion-mnist":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()

            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_test.astype('float32')
            self.x_train /= 255
            self.x_test /= 255

        if dataset == "mnist-background-images":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/mnist-background-images/mnist_background_images_train.amat")
            test = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/mnist-background-images/mnist_background_images_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "mnist-rotated-digits":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/mnist-rotated-digits/mnist_all_rotation_normalized_float_train_valid.amat")
            test = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/mnist-rotated-digits/mnist_all_rotation_normalized_float_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "mnist-random-background":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/mnist-random-background/mnist_background_random_train.amat")
            test = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/mnist-random-background/mnist_background_random_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "mnist-rotated-with-background":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/mnist-rotated-with-background/mnist_all_background_images_rotation_normalized_train_valid.amat")
            test = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/mnist-rotated-with-background/mnist_all_background_images_rotation_normalized_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "rectangles":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/rectangles/rectangles_train.amat")
            test = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/rectangles/rectangles_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "rectangles-images":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/rectangles-images/rectangles_im_train.amat")
            test = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/rectangles-images/rectangles_im_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "convex":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/convex/convex_train.amat")
            test = np.loadtxt("/content/drive/MyDrive/pso_proposed/dataset/convex/convex_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], input_channels)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], input_channels)

        self.y_train = np_utils.to_categorical(self.y_train, output_dim)
        self.y_test = np_utils.to_categorical(self.y_test, output_dim)

        print("Initializing population...")
        self.population = Population(pop_size, input_width, input_height, input_channels, max_conv_kernel, max_out_ch, max_pool_kernel, max_fc_neurons, output_dim)
                  
        print("Verifying accuracy of the current gBest...")
        print(self.population.particle[0])
        print( 'vel = '+ str(self.population.particle[0].vel))
        self.gBest_index = 0
        
        self.gBest = deepcopy(self.population.particle[0])
        self.population.particle[0].parameters = self.gBest.model_compile(dropout_rate)
        print('number of parameters: ' + str(self.population.particle[0].parameters))
        hist = self.gBest.model_fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
        self.gBest.model.save(self.results_path + "particle " + str(0) + ".h5")
        test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)
        self.gBest.model_delete()
        self.population.particle[0].acc_train = hist.history['accuracy'][-1]
        self.gBest_acc[0] = hist.history['accuracy'][-1]
        self.gBest_test_acc[0] = test_metrics[1]

        #
        self.gBest_i.append(deepcopy(self.population.particle[0]))
        self.num_gBest = self.num_gBest + 1
        self.gBest_pso_metric[0, 0] = 0  # iter
        self.gBest_pso_metric[0, 1] = test_metrics[1]  # acc
        self.gBest_pso_metric[0, 2] = test_metrics[0]  # loss
        #
        
        # self.population.particle[0].acc = hist.history['accuracy'][-1]
        # self.population.particle[0].pBest.acc = hist.history['accuracy'][-1]
        self.population.particle[0].acc = test_metrics[1]
        self.population.particle[0].pBest.acc = test_metrics[1]
        
        print("Current gBest acc: " + str(self.gBest_acc[0]) + "\n")
        print("Current gBest test acc: " + str(self.gBest_test_acc[0]) + "\n")

        print("Looking for a new gBest in the population...")
        gBest_update = 0
        for i in range(1, self.pop_size):
            print('_______________________ Initialization - Particle: ' + str(i) +'_______________________')
            print(self.population.particle[i])
            print( 'vel = '+ str(self.population.particle[i].vel))
            self.population.particle[i].model_compile(dropout_rate)
            print('number of parameters: ' + str(self.population.particle[i].parameters))
            hist = self.population.particle[i].model_fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
            self.population.particle[i].model.save(self.results_path + "particle " + str(i) + ".h5")
            #
            test_metrics = self.population.particle[i].model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)
            # 
            self.population.particle[i].model_delete()
           
            # self.population.particle[i].acc = hist.history['accuracy'][-1]
            # self.population.particle[i].pBest.acc = hist.history['accuracy'][-1]
            self.population.particle[i].acc_train = hist.history['accuracy'][-1]
            self.population.particle[i].acc = test_metrics[1]
            self.population.particle[i].pBest.acc = test_metrics[1]
            
            if self.population.particle[i].pBest.acc >= self.gBest_test_acc[0]:
                print("GGGGGGGGGGGGGGGGGGGGGGGGGGG  Found a new gBest.  GGGGGGGGGGGGGGGGGGGGGGGGGGG")
                self.gBest = deepcopy(self.population.particle[i])
                self.gBest_index = i
                #
                gBest_update = i
                if gBest_update > 0:
                  self.gBest_i[0] = deepcopy(self.population.particle[i])
                  self.gBest_pso_metric[0, 0] = 0  # iter
                  self.gBest_pso_metric[0, 1] = self.population.particle[i].acc  # acc 
                  self.gBest_pso_metric[0, 2] = test_metrics[0] #loss
                # 
                self.gBest_test_acc[0] = self.population.particle[i].pBest.acc
                print("New gBest acc(base test): " + str(self.gBest_test_acc[0]))
                
                # self.gBest.model_compile(dropout_rate)
                # test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)
                # self.gBest_test_acc[0] = test_metrics[1]
                # print("New gBest test acc: " + str(self.gBest_test_acc[0]))
            
            self.gBest.model_delete()
        print("gbest index :" + str(self.gBest_index))

    def fit(self, dropout_rate):
        for i in range(1, self.n_iter):            
            gBest_acc = self.gBest_acc[i-1]
            gBest_test_acc = self.gBest_test_acc[i-1]
            gBest_update = -1
            for j in range(self.pop_size):
                print('_______________________ Iteration: ' + str(i) + ' - Particle: ' + str(j) + '_______________________')

                # Update particle velocity
                self.population.particle[j].velocity(self.gBest.layers)
                print('vel = ' +str(self.population.particle[j].vel))
                # Update particle architecture
                self.population.particle[j].update()
                # self.population.particle[j].layers = self.population.particle[j].vel

                print('Particle NEW architecture: ')
                print(self.population.particle[j])
                # print(self.population.particle[j].layers)
                if (j != self.gBest_index):
                # Compute the acc in the updated particle
                    self.population.particle[j].model_compile(dropout_rate)
                    print('number of parameters: ' + str(self.population.particle[j].parameters))
                    hist = self.population.particle[j].model_fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs)
                
                    self.population.particle[j].model.save(self.results_path + "particle " + str(j) + ".h5")

                #
                    test_metrics = self.population.particle[j].model.evaluate(x=self.x_test, y=self.y_test, batch_size=self.batch_size)
                #
                    print("acc test : " + str(test_metrics[1]))
                    self.population.particle[j].model_delete()

                    self.population.particle[j].acc = test_metrics[1]
                    self.population.particle[j].acc_train = hist.history['accuracy'][-1]
                
                    f_test = self.population.particle[j].acc
                    pBest_acc = self.population.particle[j].pBest.acc

                    if f_test >= pBest_acc:
                        print("Found a new pBest.")
                        print("Current acc: " + str(f_test))
                        print("Past pBest acc: " + str(pBest_acc))
                        pBest_acc = f_test
                        self.population.particle[j].pBest = deepcopy(self.population.particle[j])

                        if pBest_acc >= gBest_test_acc:
                           print("GGGGGGGGGGGGGGGGGGGGGGGGGGG   Found a new gBest. GGGGGGGGGGGGGGGGGGGGGGGGGGG")
                           gBest_test_acc = pBest_acc
                           self.gBest = deepcopy(self.population.particle[j])
                           self.gBest_index = j
                           print("New gBest acc(base test): " + str(pBest_acc))
                           gBest_acc = self.population.particle[j].acc_train
                           gBest_update = j
                
            if gBest_update == -1:
                self.gBest_pso_metric[i, 0] = i
                self.gBest_pso_metric[i, 1] = gBest_test_acc
                self.gBest_pso_metric[i, 2] = self.gBest_pso_metric[i-1, 2]
            else:
                self.gBest_i.append(deepcopy(self.population.particle[gBest_update]))
                # self.population.particle[gBest_update].model_compile(dropout_rate)
                # self.population.particle[gBest_update].model.save('/content/drive/MyDrive/save/' + "gBest[" + str(self.num_gBest + 1) + "].h5")
                self.gBest_pso_metric[i, 0] = i
                self.gBest_pso_metric[i, 1] = gBest_test_acc 
                self.gBest_pso_metric[i, 2] = test_metrics[0] 
                self.num_gBest = self.num_gBest + 1
                
            self.gBest_acc[i] = gBest_acc
            self.gBest_test_acc[i] = gBest_test_acc
            print("gbest index :" + str(self.gBest_index))
            print("Current gBest acc: " + str(self.gBest_acc[i]))
            print("Current gBest test acc: " + str(self.gBest_test_acc[i]))

    def fit_gBest(self, batch_size, epochs, dropout_rate):
        print("\nFurther training gBest model...")
        self.gBest.model_compile(dropout_rate)

        trainable_count = 0
        for i in range(len(self.gBest.model.trainable_weights)):
            trainable_count += keras.backend.count_params(self.gBest.model.trainable_weights[i])
            
        print("gBest's number of trainable parameters: " + str(trainable_count))
        self.gBest.model_fit_complete(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)

        return trainable_count
    
    def evaluate_gBest(self, batch_size):
        print("\nEvaluating gBest model on the test set...")
        
        metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)

        print("\ngBest model loss in the test set: " + str(metrics[0]) + " - Test set accuracy: " + str(metrics[1]))
        return metrics

    def fit_gBesti(self, gb, batch_size, epochs, dropout_rate):
        gb.model_compile(dropout_rate)
        parameter_count = int(np.sum([keras.backend.count_params(p) for p in set(gb.model.trainable_weights)]))
        print("gBest's number of parameters: " + str(parameter_count))
        gb.model_fit_complete(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
        return parameter_count
        
    def evaluate_particle(self, particle, batch_size):
        print("\nEvaluating particle model on the test set...")
        metrics = particle.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)
        print("\n model loss in the test set: " + str(metrics[0]) + " - Test set accuracy: " + str(metrics[1]))
        return metrics
