from .misc import build_file, current_time
from .registry import HOOKS, build_from_cfg 
import time 
import os
from tensorboardX import SummaryWriter
from occFacto.config.config import get_cfg

@HOOKS.register_module()
class TextLogger:
    """
    A logger for writing text logs to a file.

    This logger creates a log file in a specified directory and appends log messages to it.

    Attributes:
    - log_file (file object): The file object for the log file.

    Parameters:
    - work_dir (str): The directory where the log file will be created.
    """

    def __init__(self,work_dir):
        """
        Initialize the TextLogger with a log file in the specified directory.
        """
        save_file = build_file(work_dir,prefix="textlog/log_"+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+".txt")
        self.log_file = open(save_file,"a")

    def log(self,data):
        """
        Write a log message to the log file.

        Parameters:
        - data (dict): A dictionary containing the data to be logged.
        """
        msg = ",".join([f"{k}:{d}" for k,d in data.items()])
        msg= current_time()+msg+"\n"
        self.log_file.write(msg)
        self.log_file.flush()


@HOOKS.register_module()
class TensorboardLogger:
    """
    A logger for recording metrics to TensorBoard.

    This logger initializes a TensorBoard writer and writes scalar values to it, which is useful for tracking and visualizing metrics over time.

    Attributes:
    - writer (SummaryWriter): The TensorBoard SummaryWriter object.

    Parameters:
    - work_dir (str): The directory where TensorBoard logs will be stored.
    """

    def __init__(self,work_dir):
        """
        Initialize the TensorboardLogger with a specified directory for TensorBoard logs.
        """
        self.cfg = get_cfg()
        tensorboard_dir = os.path.join(work_dir,"tensorboard")
        self.writer = SummaryWriter(tensorboard_dir,flush_secs=10)

    def log(self,data):
        """
        Log metrics to TensorBoard.

        Parameters:
        - data (dict): A dictionary containing the data to be logged. The 'iter' key is used as the step for TensorBoard.
        """
        step = data["iter"]
        for k,d in data.items():
            if k in ["iter","epoch","batch_idx","times","batch_size"]:
                continue
            if isinstance(d,str):
                continue
            self.writer.add_scalar(k,d,global_step=step)


@HOOKS.register_module()
class RunLogger:
    """
    A composite logger that manages multiple logging methods like TextLogger and TensorboardLogger.

    It forwards log messages to each logger and also handles formatted printing of logs.

    Attributes:
    - loggers (list): A list of logger instances.

    Parameters:
    - work_dir (str): The base directory for logging.
    - loggers (list of str): A list of logger names to be initialized and used for logging.
    """
     
    def __init__(self,work_dir,loggers=["TextLogger","TensorboardLogger"]):
        """
        Initialize the RunLogger with specified loggers in the given work directory.
        """
        self.loggers = [build_from_cfg(l,HOOKS,work_dir=work_dir) for l in loggers]
    
    def log(self,data,**kwargs):
        """
        Log data using all configured loggers and print the log.

        Parameters:
        - data (dict): A dictionary containing the data to be logged.
        - **kwargs: Additional data to be logged.
        """
        data.update(kwargs)
        data = {k:d.item() if hasattr(d,"item") else d for k,d in data.items()}
        for logger in self.loggers:
            logger.log(data)
        self.print_log(data)
    
    def get_time(self, s):
        """
        Convert a time duration in seconds into a formatted string.

        Parameters:
        - s (int or float): The time duration in seconds.

        Returns:
        - str: A string representing the time duration in [days:D, hours:H, minutes:M, seconds:S] format.
        """
        s = int(s)
        days = s // 60 // 60 // 24
        hours = s // 60 // 60 % 24
        minutes = s // 60 % 60
        seconds = s % 60
        return f' [{days}D:{hours}H:{minutes}M:{seconds}S] '

    def print_log(self,msg):
        """
        Format and print a log message.

        Parameters:
        - msg (dict or str): The message to be logged. If a dict, each key-value pair is formatted and printed.
        """
        if isinstance(msg,dict):
            msgs = []
            for k,d in msg.items():
                if (k == "remain_time"):
                    msgs.append(f" {k}:{self.get_time(d)}")
                else:
                    msgs.append(f" {k}:{d:.7f}" if isinstance(d,float) else f" {k}:{d}")
            msg = ",".join(msgs)
        print(current_time(),msg)