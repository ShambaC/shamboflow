"""Some common models"""

import numpy as np
import cupy as cp

from shamboflow import IS_CUDA
from shamboflow.engine.base_models import BaseModel
from shamboflow.engine import losses

from tqdm import tqdm, trange
from colorama import Fore

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
            print("Building layers")

        with tqdm(self.layers) as pbar :
            for layer in pbar :
                layer.build()
                self.parameters += layer.size
                if verbose :
                    pbar.update()

        if verbose :
            print(Fore.GREEN + "Finished building layers!")
            print("Generating weight matrices")
            
        for i in trange(len(self.layers) - 1) :
            size_a = self.layers[i].size
            size_b = self.layers[i+1].size

            self.parameters += size_a * size_b

            if IS_CUDA :
                weigth_mat = cp.random.uniform(-0.5, 0.5, (size_a, size_b))
                self.weights.append(weigth_mat)
            else :
                weigth_mat = np.random.uniform(-0.5, 0.5, (size_a, size_b))
                self.weights.append(weigth_mat)

        if verbose :
            print(Fore.GREEN + "Finished generating weight matrices")

        print(Fore.CYAN + "Model successfully compiled")

        self.is_compiled = True