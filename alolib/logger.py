import os 
import logging
import logging.config
from datetime import datetime
import inspect
import colorama
from colorama import Fore, Style
colorama.init()

class ColoredFormatter(logging.Formatter):
    COLORS = {
        #logging.DEBUG: Fore.GRAY,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.BLUE,
        logging.ERROR: Fore.MAGENTA,
        #logging.CRITICAL: Fore.RED # + Style.BRIGHT
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        message = super().format(record)
        return f"{log_color}{message}{Style.RESET_ALL}"
    
#--------------------------------------------------------------------------------------------------------------------------
#    Logger Class : (asset.py 및 사용자 asset에서 사용)
#--------------------------------------------------------------------------------------------------------------------------
def log_decorator(func):
    def wrapper(*args, **kwargs):
        caller_frame = inspect.stack()[1]
        # FIXME class name 찾는법 복잡해서 일단 제거 
        #caller_name = caller_frame.
        if caller_frame.function in ['save_info', 'save_warning', 'save_error']: 
            caller_frame = inspect.stack()[2] # user asset 사용자 API 호출 시엔 원본 asset 위치 알아내도록 2 depth 뒤로
        caller_func = caller_frame.function
        caller_file = os.path.basename(caller_frame.filename)
        caller_line = caller_frame.lineno
        # 원본 함수 호출
        logger_method, msg = func(*args, **kwargs)
        logger_method(f'{caller_file}({caller_line})|{caller_func}()] {msg}')
        if logger_method.__name__ == "error":
            raise
    return wrapper

def custom_log_decorator(func):
    ''' for custom log with level as integer
    '''
    def wrapper(*args, **kwargs):
        caller_frame = inspect.stack()[1]
        caller_file = os.path.basename(caller_frame.filename)
        caller_line = caller_frame.lineno
        caller_func = caller_frame.function
        # 원본 함수 호출
        logger_method, msg, level = func(*args, **kwargs)
        logger_method(msg = f'{caller_file}({caller_line})|{caller_func}()] {msg}', level = level)
        if logger_method.__name__ == "error":
            raise
    return wrapper

class Logger: 
    def __init__(self, envs, service):
        try:
            MSG_LOG_LEVEL = 11
            SHOW_LOG_LEVEL = 12
            logging.addLevelName(MSG_LOG_LEVEL, 'MSG')
            logging.addLevelName(SHOW_LOG_LEVEL, 'SHOW')
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
                    "level": "MSG", # 최소 LEVEL 
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": self.log_file_path, 
                    "formatter": "asset_file",
                    "level": "DEBUG",
                },
            },
            "root": {"handlers": ["console", "file"], "level": "DEBUG"},
            "loggers": {"ERROR": {"level": "ERROR"}, "WARNING": {"level": "WARNING"}, "INFO": {"level": "INFO"}, \
                "MSG": {"level": MSG_LOG_LEVEL}, "SHOW": {"level": SHOW_LOG_LEVEL}}
        }
        
    #--------------------------------------------------------------------------------------------------------------------------
    #    ALOlib asset & UserAsset Logging
    #--------------------------------------------------------------------------------------------------------------------------
    @custom_log_decorator
    def asset_message(self, msg):
        '''custom logging: level MSG_LOG_LEVEL(11)
        used for ALO process logging 
        ''' 
        logging.config.dictConfig(self.asset_logging_config)
        message_logger = logging.getLogger("MSG") 
        level = message_logger.level
        return message_logger.log, msg, level
    
    @custom_log_decorator
    def asset_show(self, msg): 
        '''show level (12)은 
        pipeline run 마지막 부에 table화 하여 print
        '''
        if not isinstance(msg, str):
            self.asset_error("Failed to run asset_show(). Only support << str >> type for the argument.")
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        show_logger = logging.getLogger("SHOW") 
        level = show_logger.level
        return show_logger.log, msg, level
    
    @log_decorator
    def asset_info(self, msg): 
        # UserAsset API에서도 쓰므로 str type check 필요  
        if not isinstance(msg, str):
            self.asset_error("Failed to run asset_info(). Only support << str >> type for the argument.")
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        info_logger = logging.getLogger("INFO") 
        return info_logger.info, msg 

    
    @log_decorator
    def asset_warning(self, msg):
        if not isinstance(msg, str):
            self.asset_error("Failed to run asset_warning(). Only support << str >> type for the argument.")
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        warning_logger = logging.getLogger("WARNING") 
        return warning_logger.warning, msg 
    
    @log_decorator
    def asset_error(self, msg):
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        if not isinstance(msg, str):
            msg = "Failed to run asset_error(). Only support << str >> type for the argument."
            logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
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
        # log file save 
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        error_logger = logging.getLogger("ERROR") 
        return error_logger.error, formatted_msg