"""A collection of common callback functions"""

from colorama import Fore, Back, Style

from shamboflow.engine.base_callback import BaseCallback
from shamboflow.engine.base_models import BaseModel

class EarlyStopping(BaseCallback) :
    """Early Stopper

    This callback method monitors a given
    metric and then stops the training
    early if the metric doesn't improve
    for a given amount of time
    
    """

    def __init__(self, monitor : str = 'loss', patience : int = 10, verbose : bool = False, **kwargs) -> None :
        """Initialize

        Args
        ----
            monitor : str
                The metric to monitor. It is one of the 4: `loss`, `acc`, `val_loss`, `val_acc`
            patience : int
                How many epoch to monitor before stopping
            verbose : bool = False
                Log callback function logs
        
        """

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose

        self.metric_old = 0.0
        self.patience_ctr = 0

    def run(self, model : BaseModel) -> None:
        """The callback method that will be called after each epoch"""

        if self.monitor in ('val_loss', 'val_acc') :
            if not model.has_validation_data :
                self.monitor = self.monitor.removeprefix('val_')

        current_metric = model.metrics[self.monitor]
        if model.current_epoch == 0 :
            self.metric_old = current_metric
            return
        
        if 'loss' in self.monitor :
            if current_metric >= self.metric_old :
                self.patience_ctr += 1

                if self.verbose :
                    print(f"[EarlyStopping] : {self.monitor} has not improved for the past {self.patience_ctr} epochs.")

                if self.patience_ctr == self.patience :
                    print(Back.RED + Fore.WHITE + f"Early Stopping, {self.monitor} has not improved for past {self.patience} epochs")
                    print(Style.RESET_ALL)
                    model.stop()
            else :
                self.patience_ctr = 0
        else :
            if current_metric <= self.metric_old :
                self.patience_ctr += 1

                if self.verbose :
                    print(f"[EarlyStopping] : {self.monitor} has not improved for the past {self.patience_ctr} epochs.")
                    
                if self.patience_ctr == self.patience :
                    print(Back.RED + Fore.WHITE + f"Early Stopping, {self.monitor} has not improved for past {self.patience} epochs")
                    print(Style.RESET_ALL)
                    model.stop()
            else :
                self.patience_ctr = 0

        self.metric_old = current_metric


class ReduceLROnPlateau(BaseCallback) :
    """Reduce Learning Rate on Plateau callback
    
    This callback function monitors a
    given metric for a set amount of
    iterations and reduces the learning
    rate if the metric doesn't improve
    for the given duration.
    """

    def __init__(self, monitor : str = 'loss', patience : int = 10, factor : float = 0.9, min_val : float = 1e-10, verbose : bool = False, **kwargs) -> None :
        """Initialize
        
        Args
        ----
            monitor : str
                The metric to monitor. It is one of the 4: `loss`, `acc`, `val_loss`, `val_acc`
            patience : int
                How many epoch to monitor before stopping
            factor : float = 0.9
                fraction to which the learning rate will be lowered to. Note - This is not by how much to reduce but how much to reduce to
            min_val : float = 1e-10
                Min possible learning rate
            verbose : bool = False
                Log callback function logs
        """

        self.monitor = monitor
        self.patience = patience
        self.factor = factor
        self.min_val = min_val
        self.verbose = verbose

        self.metric_old = 0.0
        self.patience_ctr = 0

    def run(self, model : BaseModel) -> None:
        """The callback method that will be called after each epoch"""

        if self.monitor in ('val_loss', 'val_acc') :
            if not model.has_validation_data :
                self.monitor = self.monitor.removeprefix('val_')

        current_metric = model.metrics[self.monitor]
        if model.current_epoch == 0 :
            self.metric_old = current_metric
            return
        
        if 'loss' in self.monitor :
            if current_metric >= self.metric_old :
                self.patience_ctr += 1

                if self.verbose :
                    print(f"[ReduceLROnPlateau] : {self.monitor} has not improved for the past {self.patience_ctr} epochs.")

                if self.patience_ctr == self.patience :
                    new_LR = self.factor * model.learning_rate
                    model.learning_rate = new_LR if new_LR > self.min_val else model.learning_rate

                    print(Back.RED + Fore.WHITE + f"Reducing Learning Rate, {self.monitor} has not improved for past {self.patience} epochs. New Learning rate is {model.learning_rate}")
                    print(Style.RESET_ALL)
            else :
                self.patience_ctr = 0
        else :
            if current_metric <= self.metric_old :
                self.patience_ctr += 1

                if self.verbose :
                    print(f"[ReduceLROnPlateau] : {self.monitor} has not improved for the past {self.patience_ctr} epochs.")
                    
                if self.patience_ctr == self.patience :
                    new_LR = self.factor * model.learning_rate
                    model.learning_rate = new_LR if new_LR > self.min_val else model.learning_rate

                    print(Back.RED + Fore.WHITE + f"Reducing Learning Rate, {self.monitor} has not improved for past {self.patience} epochs. New Learning rate is {model.learning_rate}")
                    print(Style.RESET_ALL)
            else :
                self.patience_ctr = 0

        self.metric_old = current_metric

class ModelCheckpoint(BaseCallback) :
    """Model Checkpointing
    
    This callback function monitors a
    metric and saves the instance of model
    to disk at every epoch or on a condition
    """

    def __init__(self, save_path : str = "model.ckpt", monitor : str = 'loss', save_best_only : bool = True, verbose : bool = False, **kwargs) -> None :
        """Initialize
        
        Args
        ----
            save_path : str = 'model.ckpt'
                Save path for the model
            monitor : str
                The metric to monitor. It is one of the 4: `loss`, `acc`, `val_loss`, `val_acc`
            save_best_only : bool = True
                whether to save only the best model according to the monitor
            verbose : bool = False
                Log callback function logs

        """

        self.save_path = save_path
        self.monitor = monitor
        self.save_best = save_best_only
        self.verbose = verbose

        self.metric_old = 0.0

    def run(self, model : BaseModel) -> None:
        """The callback method that will be called after each epoch"""

        if not self.save_best :
            if self.verbose :
                print(f"[ModelCheckpoint] : Saving to {self.save_path}")
            model.save(self.save_path)
            return

        if self.monitor in ('val_loss', 'val_acc') :
            if not model.has_validation_data :
                self.monitor = self.monitor.removeprefix('val_')

        current_metric = model.metrics[self.monitor]
        if model.current_epoch == 0 :
            self.metric_old = current_metric
            return
        
        if 'loss' in self.monitor :
            if current_metric < self.metric_old :
                if self.verbose :
                    print(f"[ModelCheckpoint] : Metric has improved. Saving to {self.save_path}")
                model.save(self.save_path)
        else :
            if current_metric > self.metric_old :
                if self.verbose :
                    print(f"[ModelCheckpoint] : Metric has improved. Saving to {self.save_path}")
                model.save(self.save_path)

        self.metric_old = current_metric