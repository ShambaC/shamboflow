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
                    print(f"{self.monitor} has not improved for the past {self.patience_ctr} epochs.")

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
                    print(f"{self.monitor} has not improved for the past {self.patience_ctr} epochs.")
                    
                if self.patience_ctr == self.patience :
                    print(Back.RED + Fore.WHITE + f"Early Stopping, {self.monitor} has not improved for past {self.patience} epochs")
                    print(Style.RESET_ALL)
                    model.stop()
            else :
                self.patience_ctr = 0

        self.metric_old = current_metric


class ReduceLROnPlateau(BaseCallback) :
    """Reduce Learning Rate on Plateau callback
    

    """