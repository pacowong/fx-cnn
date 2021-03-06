from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from utilities.stats.logger import Logger
from utilities.stats.individual_stat import stats
from utilities.fitness.image_processor import ImageProcessor
from utilities.fitness.network_processor import NetworkProcessor
from utilities.fitness.network import Network, RegressionNet, ClassificationNet
from utilities.fitness.preprocess import DataIterator, check_class_balance, read_cifar
from utilities.fitness.read_xy import DataReader
from sklearn.model_selection import train_test_split, KFold
import cv2 as cv
import numpy as np
import os, csv, random, pickle, time

class cifar10(base_ff):
    maximise = params['MAXIMIZE']  # True as it ever was.
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.fcn_layers = params['FCN_LAYERS']
        self.conv_layers = params['CONV_LAYERS']
        self.resize = params['RESIZE']

        # Read images from dataset
        X, y = DataReader.read_data(params['DATASET_ID'])

        # Train & test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        if params['NORMALIZE_LABEL']:
            Logger.log("Normalizing labels...")
            print(self.y_train, self.y_test)
            self.y_train, mean, std = ImageProcessor.normalize(self.y_train)
            self.y_test, _, _ = ImageProcessor.normalize(self.y_test, mean=mean, std=std)
            print(self.y_train, self.y_test)
            Logger.log("Mean / Std of training set (by channel): {} / {}".format(mean, std))

        # Check class balance between splits
        classes, class_balance_train, class_balance_test = check_class_balance(self.y_train, self.y_test)
        Logger.log("---------------------------------------------------", info=False)
        Logger.log("Class Balance --", info=False)
        Logger.log("\tClass: \t{}".format("\t".join([str(c) for c in classes])), info=False)
        Logger.log("\tTrain: \t{}".format("\t".join([str(n) for n in class_balance_train])), info=False)
        Logger.log("\tTest: \t{}".format("\t".join([str(n) for n in class_balance_test])), info=False)
        Logger.log("\tTotal: \t{}\t{}".format("\t".join([str(n) for n in class_balance_train + class_balance_test]), (class_balance_train + class_balance_test).sum()), info=False)

        Logger.log("---------------------------------------------------", info=False)
        Logger.log("General Setup --", info=False)
        Logger.log("\tCUDA enabled: \t{}".format(params['CUDA_ENABLED']), info=False)
        Logger.log("\tDebug network enabled: \t{}".format(params['DEBUG_NET']), info=False)

        Logger.log("---------------------------------------------------", info=False)
        Logger.log("Data Preprocess --", info=False)
        Logger.log("\tNumber of samples: \t{}".format(len(X)), info=False)
        Logger.log("\tTraining / Test split: \t{}/{}".format(len(self.X_train), len(self.X_test)), info=False)
        Logger.log("\tImage size: \t{}".format(self.X_train[0].shape), info=False)
        Logger.log("\tNormalize label: \t{}".format(params['NORMALIZE_LABEL']), info=False)
        Logger.log("\tNormalize after preprocessing: \t{}".format(params['NORMALIZE']), info=False)

        Logger.log("---------------------------------------------------", info=False)
        Logger.log("GP Setup --", info=False)
        Logger.log("\tGrammar file: \t{}".format(params['GRAMMAR_FILE']), info=False)
        Logger.log("\tPoupulation size: \t{}".format(params['POPULATION_SIZE']), info=False)
        Logger.log("\tGeneration num: \t{}".format(params['GENERATIONS']), info=False)
        Logger.log("\tImage resizing (after proc): \t{}".format(self.resize), info=False)
        Logger.log("\tTree depth init (Min/Max): \t{}/{}".format(params['MIN_INIT_TREE_DEPTH'], params['MAX_INIT_TREE_DEPTH']), info=False)
        Logger.log("\tTree depth Max: \t\t{}".format(params['MAX_TREE_DEPTH']), info=False)

    def evaluate(self, ind, **kwargs):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.

        p, d = ind.phenotype, {}

        genome, output, invalid, max_depth, nodes = ind.tree.get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(),[], [])
        Logger.log("Depth: {0}\tGenome: {1}".format(max_depth, genome))

        ## Evolve image preprocessor
        Logger.log("Processing Pipeline Start: {} images...".format(len(self.X_train)+len(self.X_test)))
        processed_train = ImageProcessor.process_images(self.X_train, ind.tree.children[0], resize=self.resize)
        processed_test = ImageProcessor.process_images(self.X_test, ind.tree.children[0], resize=self.resize)
        # TODO: Log image processing pipeline

        # Normalize image by channel
        if params['NORMALIZE']:
            Logger.log("Normalizing processed images...")
            processed_train, mean, std = ImageProcessor.normalize_img(processed_train)
            processed_test, _, _ = ImageProcessor.normalize_img(processed_test, mean=mean, std=std)
            Logger.log("Mean / Std of training set (by channel): {} / {}".format(mean, std))

        # Setup test images
        X_test, y_test = processed_test, self.y_test
        image = ImageProcessor.image
        init_size = image.shape[0]*image.shape[1]*image.shape[2]

        ## Evolve network structure
        if params['EVOLVE_NETWORK']:
            Logger.log("Network Structure Selection Start: ")
            flat_ind, new_conv_layers = NetworkProcessor.process_network(ind.tree.children[1], image.shape, self.conv_layers)
            conv_outputs = Network.calc_conv_output(new_conv_layers, image.shape)
            Logger.log("\tIndividual: {}".format(flat_ind))
            Logger.log("\tNew convolution layers: ")
            for i, a, b in zip(range(len(new_conv_layers)), new_conv_layers, conv_outputs):
                Logger.log("\tConv / output at layer {}: {}\t=> {}".format(i, a, b))
        else:
            new_conv_layers = self.conv_layers

        # Modify fully connected input size
        new_fcn_layers, conv_output = self.fcn_layers, conv_outputs[-1]
        new_fcn_layers[0] = conv_output[0]*conv_output[1]*conv_output[2]
        net = eval(params['NETWORK'])(new_fcn_layers, new_conv_layers)

        kf = KFold(n_splits=params['CROSS_VALIDATION_SPLIT'])
        fitness, fold = 0, 1

        Logger.log("Training Start: ")

        # Cross validation
        s_time = np.empty((kf.get_n_splits()))
        # validation_acc, validation_acc5 = np.empty((kf.get_n_splits())), np.empty((kf.get_n_splits()))
        # test_acc, test_acc5 = np.empty((kf.get_n_splits())), np.empty((kf.get_n_splits()))
        for train_index, val_index in kf.split(processed_train):
            X_train, X_val = processed_train[train_index], processed_train[val_index]
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]
            data_train = DataIterator(X_train, y_train, params['BATCH_SIZE'])
            early_ckpt, early_stop, early_crit, epsilon = 4, [], params['EARLY_STOP_FREQ'], params['EARLY_STOP_EPSILON']
            s_time[fold-1] = time.time()

            # Train model
            net.model.reinitialize_params()
            for epoch in range(1, params['NUM_EPOCHS'] + 1):
                # mini-batch training
                for x, y in data_train:
                    net.train(epoch, x, y)

                # log training loss
                if epoch % params['TRAIN_FREQ'] == 0:
                    Logger.log("Epoch {} Training loss (NLL): {:.6f}".format(epoch, net.train_loss.getLoss()))

                # log validation/test loss
                if epoch % params['VALIDATION_FREQ'] == 0 or epoch < 15:
                    net.test(X_val, y_val)
                    Logger.log("Epoch {} Validation loss (NLL/Accuracy): {}".format(epoch, net.get_test_loss_str()))
                    net.test(X_test, y_test)
                    Logger.log("Epoch {} Test loss (NLL/Accuracy): {}".format(epoch, net.get_test_loss_str()))

                # check for early stop
                if epoch == early_ckpt:
                    accuracy = net.test(X_test, y_test, print_confusion=True)
                    early_stop.append(accuracy)
                    if len(early_stop) > 3:
                        latest_acc = early_stop[-early_crit:]
                        latest_acc = np.subtract(latest_acc, latest_acc[1:]+[0])
                        if (abs(latest_acc[:-1]) < epsilon).all() == True:
                            Logger.log("Early stopping at epoch {} (latest {} ckpts): {}".format(epoch, early_crit, " ".join(["{:.4f}".format(x) for x in early_stop[-early_crit:]])))
                            break
                    # early_ckpt = min(early_ckpt+300, early_ckpt*2)
                    early_ckpt += params['VALIDATION_FREQ']

            # Validate model
            net.test(X_val, y_val)
            net.save_validation_loss()
            Logger.log("Cross Validation [Fold {}/{}] Validation (NLL/Accuracy): {}".format(fold, kf.get_n_splits(), net.get_test_loss_str()))

            # Test model
            net.test(processed_test, self.y_test)
            net.save_test_loss()
            Logger.log("Cross Validation [Fold {}/{}] Test (NLL/Accuracy): {}".format(fold, kf.get_n_splits(), net.get_test_loss_str()))

            # Calculate time
            s_time[fold-1] = time.time() - s_time[fold-1]
            Logger.log("Cross Validation [Fold {}/{}] Training Time (m / m per epoch): {:.3f} {:.3f}".format(fold, kf.get_n_splits(), s_time[fold-1]/60, s_time[fold-1]/60/epoch))

            fold = fold + 1

        fitness = net.get_fitness()

        val_log, test_log = np.array(net.validation_log), np.array(net.test_log)
        print(val_log, test_log)
        val_mean, test_mean = val_log.mean(axis=1), test_log.mean(axis=1)
        for i, (v, t) in enumerate(zip(val_log, test_log)):
            print(i,v,t)
            log = " ".join(["{:.4f} {:.4f}".format(v[idx], t[idx]) for idx in range(len(v))])
            Logger.log("STAT -- Model[{}/{}] #{:.3f}m Validation / Generalization: {}".format(i, kf.get_n_splits(), s_time[i]/60, log))
        Logger.log("STAT -- Mean Validation / Generatlization: {}".format(" ".join(["{:.4f} {:.4f}".format(i, j) for i, j in zip(val_mean, test_mean)])))
        # ind.net = net
        params['CURRENT_EVALUATION'] += 1
        return fitness
