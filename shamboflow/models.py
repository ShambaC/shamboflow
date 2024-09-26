"""Some common models"""

import numpy as np
import cupy as cp

from shamboflow import IS_CUDA
from shamboflow.engine.base_models import BaseModel
from shamboflow.engine import losses
from shamboflow.engine import d_losses, d_activations

from tqdm import tqdm, trange
from colorama import Fore, Back, Style

class Sequential(BaseModel) :
    """A simple sequential model with multiple layers one after the other
    
    This is the simplest model type
    that is commonly used. It has
    multiple layers in it sequentially
    connected by dense connected directed
    graph between neurons.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compile(self, loss : str, learning_rate : float = 0.001, verbose : bool = False, **kwargs) -> None:
        """Method to initialize and compile the model
        
        Initializes all the required values
        and performs required operations.

        Args
        ----
            loss : str
                The loss function to use
            learning_rate : float
                The learning_rate to use while fitting data
            verbose : bool
                Should progress be displayed in details

        """

        self.learning_rate = learning_rate
        self.loss_str = loss
        self.loss = losses.get(loss)

        if verbose :
            print(Fore.CYAN + "Compiling Model ...")
            print(Style.RESET_ALL + "Building layers")

        with tqdm(self.layers) as pbar :
            for layer in pbar :
                layer.build()
                self.parameters += layer.size
                if verbose :
                    pbar.update()

        if verbose :
            print(Fore.GREEN + "Finished building layers!")
            print(Style.RESET_ALL + "Generating weight matrices")
            
        for i in trange(len(self.layers) - 1) :
            size_a = self.layers[i].size
            size_b = self.layers[i+1].size

            self.parameters += size_a * size_b

            if IS_CUDA :
                weight_mat = cp.random.uniform(-0.5, 0.5, (size_a, size_b))
                self.weights.append(weight_mat)
            else :
                weight_mat = np.random.uniform(-0.5, 0.5, (size_a, size_b))
                self.weights.append(weight_mat)

        if verbose :
            print(Fore.GREEN + "Finished generating weight matrices")

        print(Fore.CYAN + "Model successfully compiled")
        print(Style.RESET_ALL)

        self.is_compiled = True

    def fit(self, train_x : np.ndarray, train_y : np.ndarray, epochs : int, **kwargs) -> None:
        """Method to train the model and fit the data

        It runs the training where the
        network does the learning.

        Args
        ----
            train_x : ndarray
                The features of the dataset
            train_y : ndarray
                The label of the dataset
            epochs : int
                THe number of steps to run the training for
            
        Kwargs
        ------
            validation_x : ndarray
            validation_y : ndarray
            callbacks : list
                A list of callback methods
        
        """

        self.train_data_x = train_x
        self.train_data_y = train_y
        self.epochs = epochs

        if 'validation_x' in kwargs and 'validation_y' in kwargs :
            self.has_validation_data = True
            self.validation_x = kwargs['validation_x']
            self.validation_y = kwargs['validation_y']

        if 'callbacks' in kwargs :
            self.callbacks = kwargs['callbacks']

        num_rows = self.train_data_x.shape[0]

        while self.is_fitting and self.current_epoch <= self.epochs :

            with tqdm(total=num_rows) as pbar :
                def row_iter(x) :
                    num_layer = -1

                    # Forward propagation
                    for layer in self.layers :
                        num_layer += 1
                        if num_layer == 0 :
                            layer.compute(input = x)
                            continue

                        if IS_CUDA :
                            op_gpu = cp.asarray(self.layers[num_layer - 1].output_array)
                            weight_gpu = cp.asarray(self.weights[num_layer - 1])
                            layer.compute(cp.matmul(op_gpu, weight_gpu))
                        else :
                            op = self.layers[num_layer - 1].output_array
                            weight = self.weights[num_layer - 1]
                            layer.compute(np.matmul(op, weight))
                    
                    self.error_val = self.loss(self.layers[num_layer].output_array, self.train_data_y[num_rows])
                    acc = np.subtract(self.train_data_y[num_rows], self.layers[num_layer].output_array)
                    self.accuracy_val = (self.layers[num_layer].size - np.count_nonzero(acc)) / self.layers[num_layer].size

                    # BackPropagation
                    ## Compute Gradients for output layer
                    d_loss_fun = d_losses.get(self.loss_str)
                    d_act_fun = d_activations.get(self.layers[num_layer].activation_str)


                    pbar.set_postfix_str(f"Accuracy: {self.accuracy_val}, Loss: {self.error_val}")
                    pbar.update(1)    

                np.apply_along_axis(row_iter, 0, self.train_data_x)
            
            self.current_epoch += 1
        

    def summary(self) -> None:
        """Prints a summary of the model once compiled"""

        if not self.is_compiled :
            print(Back.RED + Fore.WHITE + "Model has not been compiled.\nCompile the model first using model.compile()")
            print(Style.RESET_ALL)
            return

        print(Fore.WHITE + "Model type : " + Fore.CYAN + "Sequential\n")
        print(Fore.WHITE + "Layers: ")

        for layer in self.layers :
            print("-> " + Fore.CYAN + layer.name + Fore.WHITE + f"Neurons: {layer.size} Activation: {layer.activation_str} Trainable: {layer.trainable}")

        print(Style.RESET_ALL)
        print(f"\nTrainable Params: {self.parameters}")

    def save(self, save_path : str) -> None:
        """Method to save the model to disk
        
        Args
        ----
            save_path : str
                Path to where the model will be saved, along with the name of the model file
        """

        import pickle
        import os

        if not os.path.isfile(save_path) :
            save_path += "/model.meow"

        with open(save_path, 'wb') as f :
            pickle.dump(self, f)
            print(f"Saved model to: {save_path}")


def load_model(path_to_model : str) -> BaseModel :
    """Method to load a model from disk
    
    Args
    ----
        path_to_model : str
            path to the model file
    
    Returns
    -------
        The model object

    """

    import pickle

    with open(path_to_model, 'rb') as f :
        model = pickle.load(f)
        return model