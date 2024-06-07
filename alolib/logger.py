import os 
import logging
import logging.config
from datetime import datetime
import inspect
import colorama
from colorama import Fore, Style
from copy import deepcopy
colorama.init()

class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.BLUE,
        logging.ERROR: Fore.MAGENTA,
    }

    def format(self, record):
        """ logger color formatting

        Args:           
            record   (str): logging message
            
        Returns:
            formatted message

        """
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        message = super().format(record)
        return f"{log_color}{message}{Style.RESET_ALL}"
    
#--------------------------------------------------------------------------------------------------------------------------
#    Logger Class : (asset.py 및 사용자 asset에서 사용)
#--------------------------------------------------------------------------------------------------------------------------
def log_decorator(func):
    """ log function decorator for finding original caller 

    Args:           
        func   (function): original function tobe decorated 
        
    Returns:
        wrapper (function): wrapped function 
    
    """
    def wrapper(*args, **kwargs):
        caller_frame = inspect.stack()[1]
        if caller_frame.function in ['save_info', 'save_warning', 'save_error']: 
            ## When calling the user asset API, determine the original asset location by going two depths back.
            caller_frame = inspect.stack()[2] 
        caller_func = caller_frame.function
        caller_file = os.path.basename(caller_frame.filename)
        caller_line = caller_frame.lineno
        ## Call the original function 
        logger_method, msg = func(*args, **kwargs)
        logger_method(f'{caller_file}({caller_line})|{caller_func}()] {msg}')
        if logger_method.__name__ == "error":
            raise
    return wrapper

def custom_log_decorator(func):
    """ custom log function decorator with integer logger level

    Args:           
        func   (function): original function tobe decorated 
        
    Returns:
        wrapper (function): wrapped function 
    
    """
    def wrapper(*args, **kwargs):
        caller_frame = inspect.stack()[1]
        caller_file = os.path.basename(caller_frame.filename)
        caller_line = caller_frame.lineno
        caller_func = caller_frame.function
        ## Call the original function
        logger_method, msg, level = func(*args, **kwargs)
        ## integer logger level 
        logger_method(msg = f'{caller_file}({caller_line})|{caller_func}()] {msg}', level = level)
        if logger_method.__name__ == "error":
            raise
    return wrapper

class Logger: 
    def __init__(self, envs, service):
        """ initialize logger config

        Args:           
            envs    (dict): environmental info. from ALO master
            service (str): service name (e.g. ALO, USER)
            
        Returns: -
        
        """
        try:
            MSG_LOG_LEVEL = 11
            logging.addLevelName(MSG_LOG_LEVEL, 'MSG')
            self.asset_envs = envs
            self.init_file_name = inspect.getframeinfo(inspect.currentframe().f_back)
            self.project_home = self.asset_envs['project_home']
            self.pipeline = self.asset_envs['pipeline']
            self.step = self.asset_envs['step']
            self.log_file_path = self.asset_envs['log_file_path']
        except Exception as e: 
            raise ValueError('[LOGGER][ERROR] Logger class requires properly set argument << envs >> \n' + e, color='red') 
        self.service = service
        self.asset_logging_config = { 
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "asset_console": {
                    "()": ColoredFormatter,
                    "format": f"[%(asctime)s|{self.service}|%(levelname)s|%(message)s"
                },
                "asset_file": {
                    "format": f"[%(asctime)s|{self.service}|%(levelname)s|%(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "asset_console",
                    "level": "MSG", 
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": self.log_file_path, 
                    "formatter": "asset_file",
                    "level": "MSG",
                },
            },
            ## minimum level base: MSG
            "loggers": {"ERROR": {"level": "ERROR"}, "WARNING": {"level": "WARNING"}, "INFO": {"level": "INFO"}, "MSG": {"level": MSG_LOG_LEVEL}},
            "root": {"handlers": ["console", "file"], "level": "MSG"}
        }
        
    @custom_log_decorator
    def asset_message(self, msg):
        """ custom logging API used for ALO process logging  - level MSG_LOG_LEVEL(11)

        Args:           
            msg (str): logging message
            
        Returns: 
            logger.log
            message (str)
            logger.level
        
        """
        if not isinstance(msg, str):
            self.asset_error("Failed to run asset_message(). Only support << str >> type for the argument.")
        logging.config.dictConfig(self.asset_logging_config)
        message_logger = logging.getLogger("MSG") 
        level = message_logger.level
        return message_logger.log, msg, level
    
    
    @log_decorator
    def asset_info(self, msg, show=False):
        """ info logging API used for user asset logging

        Args:           
            msg (str): logging message
            
        Returns: 
            logger.info
            message (str)
        
        """ 
        if not isinstance(msg, str):
            self.asset_error("Failed to run asset_info(). Only support << str >> type for the argument.")
        if show==True: 
            asset_logging_config = deepcopy(self.asset_logging_config)
            asset_logging_config["formatters"]["asset_file"]["format"] = self.asset_logging_config["formatters"]["asset_file"]["format"].replace("[", "[SHOW|")
            logging.config.dictConfig(asset_logging_config) 
        else: 
            logging.config.dictConfig(self.asset_logging_config) 
        info_logger = logging.getLogger("INFO") 
        return info_logger.info, msg 

    
    @log_decorator
    def asset_warning(self, msg):
        """ warning logging API used for user asset logging

        Args:           
            msg (str): logging message
            
        Returns: 
            logger.warning
            message (str)
        
        """
        if not isinstance(msg, str):
            self.asset_error("Failed to run asset_warning(). Only support << str >> type for the argument.")
        logging.config.dictConfig(self.asset_logging_config) 
        warning_logger = logging.getLogger("WARNING") 
        return warning_logger.warning, msg 
    
    @log_decorator
    def asset_error(self, msg):
        """ error logging API used for user asset logging

        Args:           
            msg (str): logging message
            
        Returns: 
            logger.error
            message (str)
        
        """
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        if not isinstance(msg, str):
            msg = "Failed to run asset_error(). Only support << str >> type for the argument."
            logging.config.dictConfig(self.asset_logging_config) 
            error_logger = logging.getLogger("ERROR") 
            error_logger.error(f'{formatted_msg}') 
            raise
        formatted_msg = "".join([
            f"\n=========================================================== ASSET ERROR ===========================================================\n",
            f"TIME(UTC)   : {time_utc}\n",
            f"PIPELINE    : {self.pipeline}\n",
            f"STEP        : {self.step}\n",
            f"ERROR(msg)  : {msg}\n",
            f"=======================================================================================================================================\n"])
        logging.config.dictConfig(self.asset_logging_config) 
        error_logger = logging.getLogger("ERROR") 
        return error_logger.error, formatted_msg