import os 
import logging
import logging.config
from datetime import datetime
from copy import deepcopy 
from alolib.log_formatter import ColorizedArgsFormatter
#--------------------------------------------------------------------------------------------------------------------------
#    GLOBAL VARIABLE
#--------------------------------------------------------------------------------------------------------------------------
LINE_LENGTH = 120
# PROJECT_HOME = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/"
# TRAIN_LOG_PATH =  PROJECT_HOME + ".train_artifacts/log/"
# INFERENCE_LOG_PATH = PROJECT_HOME + ".inference_artifacts/log/"
# print_color related variables 
COLOR_RED = '\033[91m'
COLOR_END = '\033[0m'
ARG_NAME_MAX_LENGTH = 30
COLOR_DICT = {
   'PURPLE':'\033[95m',
   'CYAN':'\033[96m',
   'DARKCYAN':'\033[36m',
   'BLUE':'\033[94m',
   'GREEN':'\033[92m',
   'YELLOW':'\033[93m',
   'RED':'\033[91m',
   'BOLD':'\033[1m',
   'UNDERLINE':'\033[4m',
}
COLOR_END = '\033[0m'

def print_color(msg, _color):
    """ Description
        -----------
            Display text with color 

        Parameters
        -----------
            msg (str) : text
            _color (str) : PURPLE, CYAN, DARKCYAN, BLUE, GREEN, YELLOW, RED, BOLD, UNDERLINE

        example
        -----------
            print_color('Display color text', 'BLUE')
    """
    if _color.upper() in COLOR_DICT.keys():
        print(COLOR_DICT[_color.upper()] + msg + COLOR_END)
    else:
        raise ValueError('[ASSET][ERROR] print_color() function call error. - selected color : {}'.format(COLOR_DICT.keys()))
    
#--------------------------------------------------------------------------------------------------------------------------
#    ProcessLogger Class : (ALO master에서만 사용)
#--------------------------------------------------------------------------------------------------------------------------
# TODO https://betterstack.com/community/questions/how-to-color-python-logging-output/ 등 참고하여 log config의 stream handler에 색깔 넣는 방법 있다면 현재 print_color로 출력후 file handler만 쓴는 방식 수정하는게 좋을듯 
class ProcessLogger: 
    # [%(filename)s:%(lineno)d]
    # envs 미입력 시 설치 과정, 프로세스 진행 과정 등 전체 과정 보기 위한 로그 생성 
    def __init__(self, project_home: str):
        try: 
            self.project_home = project_home
        except: 
            print_color("[LOGGER][ERROR] Argument << project_home: str >> required for initializing ProcessLogger.", color='red')
        self.train_log_path = self.project_home + ".train_artifacts/log/"
        self.inference_log_path = self.project_home + ".inference_artifacts/log/"
        if not os.path.exists(self.train_log_path):
            os.makedirs(self.train_log_path)
        if not os.path.exists(self.inference_log_path):
            os.makedirs(self.inference_log_path)
        # 현재 pipeline 등 환경 정보를 알기 전에 큼직한 단위로 install 과정 등에 대한 logging을 alo master에서 진행 가능하도록 config
        # custom formatter: https://code.djangoproject.com/ticket/15749
        self.process_logging_config = { 
            "version": 1,
            "formatters": {
                "proc_console": {
                    "()": ColorizedArgsFormatter,
                    "format": f"[%(levelname)s][PROCESS][%(asctime)s]: %(message)s" # [%(filename)s:%(lineno)d]
                },
                "meta_console": {
                    "()": ColorizedArgsFormatter,
                    "format": f"[%(levelname)s][META][%(asctime)s]: %(message)s"
                },
                "proc_file": {
                    "format": f"[%(levelname)s][PROCESS][%(asctime)s]: %(message)s"
                },
                "meta_file": {
                    "format": f"[%(levelname)s][META][%(asctime)s]: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "proc_console",
                    "level": "INFO",
                },
                "file_train": {
                    "class": "logging.FileHandler",
                    "filename": self.train_log_path + "process.log", 
                    "formatter": "proc_file",
                    "level": "INFO",
                },
                "file_inference": {
                    "class": "logging.FileHandler",
                    "filename": self.inference_log_path + "process.log", 
                    "formatter": "proc_file",
                    "level": "INFO",
                },
            },
            #"root": {"handlers": ["console", "file_train", "file_inference"], "level": "INFO"},
            "root": {"handlers": ["file_train", "file_inference"], "level": "INFO"},
            "loggers": {"ERROR": {"level": "ERROR"}, "WARNING": {"level": "WARNING"}, "INFO": {"level": "INFO"}}
        }
        self.meta_logging_config = deepcopy(self.process_logging_config)
        self.meta_logging_config["handlers"]["console"]["formatter"] = "meta_console"
        self.meta_logging_config["handlers"]["file_train"]["formatter"] = "meta_file"
        self.meta_logging_config["handlers"]["file_inference"]["formatter"] = "meta_file"


    #--------------------------------------------------------------------------------------------------------------------------
    #    Process Logging API
    #--------------------------------------------------------------------------------------------------------------------------
    # https://velog.io/@qlgks1/python-python-logging-%ED%95%B4%EB%B6%80
    # https://www.daleseo.com/python-logging-config/
    # process 로깅은 alo master에서만 쓰므로, 굳이 str type check 안함 
    def process_meta(self, msg): 
        logging.config.dictConfig(self.meta_logging_config)
        meta_logger = logging.getLogger("INFO")
        meta_logger.info(f'{msg}')
        
        
    def process_info(self, msg):
        logging.config.dictConfig(self.process_logging_config)
        info_logger = logging.getLogger("INFO") 
        info_logger.info(f'{msg}')


    def process_warning(self, msg):
        logging.config.dictConfig(self.process_logging_config)
        warning_logger = logging.getLogger("WARNING") 
        warning_logger.warning(f'{msg}')


    def process_error(self, msg):
        logging.config.dictConfig(self.process_logging_config)
        error_logger = logging.getLogger("ERROR") 
        error_logger.error(f'{msg}') #, stack_info=True, exc_info=True)
        raise

    
class Logger: 
    def __init__(self, envs):
        try:
            self.asset_envs = envs
            self.project_home = self.asset_envs['project_home']
            self.pipeline = self.asset_envs['pipeline']
            self.step = self.asset_envs['step']
            self.log_file_path = self.asset_envs['log_file_path']
        except Exception as e: 
            print_color('[LOGGER][ERROR] Logger class requires properly set argument << envs >>. \n' + e, color='red') 
        
        self.asset_logging_config = { 
            "version": 1,
            "formatters": {
                "asset_console": {
                    "()": ColorizedArgsFormatter,
                    "format": f"[%(levelname)s][ASSET][%(asctime)s][{self.pipeline}][{self.step}]: %(message)s"
                },
                "asset_file": {
                    "format": f"[%(levelname)s][ASSET][%(asctime)s][{self.pipeline}][{self.step}]: %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "asset_console",
                    "level": "INFO",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": self.log_file_path, 
                    "formatter": "asset_file",
                    "level": "INFO",
                },
            },
            "root": {"handlers": ["console", "file"], "level": "INFO"},
            "loggers": {"ERROR": {"level": "ERROR"}, "WARNING": {"level": "WARNING"}, "INFO": {"level": "INFO"}},
        }
        
   
    
    #--------------------------------------------------------------------------------------------------------------------------
    #    ALOlib asset & UserAsset Logging
    #--------------------------------------------------------------------------------------------------------------------------
     
    def asset_info(self, msg, msg_loc=None): 
        # UserAsset API에서도 쓰므로 str type check 필요  
        if not isinstance(msg, str):
            self.asset_error("Failed to run asset_info(). Only support << str >> type for the argument.")
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        info_logger = logging.getLogger("INFO") 
        if msg_loc is not None: 
            msg = f"[{msg_loc}]" + msg
        info_logger.info(f'{msg}')

   
    def asset_warning(self, msg, msg_loc=None):
        if not isinstance(msg, str):
            self.asset_error("Failed to run asset_warning(). Only support << str >> type for the argument.")
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        warning_logger = logging.getLogger("WARNING") 
        if msg_loc is not None: 
            msg = f"[{msg_loc}]" + msg
        warning_logger.warning(f'{msg}')
    
    
    def asset_error(self, msg, msg_loc=None):
        # stdout 
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        if not isinstance(msg, str):
            msg = "Failed to run asset_error(). Only support << str >> type for the argument."
            logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
            error_logger = logging.getLogger("ERROR") 
            error_logger.error(f'{formatted_msg}') #, stack_info=True, exc_info=True)
            raise
        if msg_loc is None:     
            formatted_msg = "".join([
                f"\n\n============================= ASSET ERROR =============================\n",
                f"TIME(UTC)   : {time_utc}\n",
                f"PIPELINE    : {self.pipeline}\n",
                f"STEP        : {self.step}\n",
                f"ERROR(msg)  : {msg}\n",
                f"=======================================================================\n\n"])
        else: 
            formatted_msg = "".join([
                f"\n\n============================= ASSET ERROR =============================\n",
                f"TIME(UTC)   : {time_utc}\n",
                f"PIPELINE    : {self.pipeline}\n",
                f"STEP        : {self.step}\n",
                f"ERROR(msg)  : {msg}\n",
                f"ERROR(loc)  : {msg_loc}\n",
                f"=======================================================================\n\n"])
        # log file save 
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        error_logger = logging.getLogger("ERROR") 
        error_logger.error(f'{formatted_msg}') #, stack_info=True, exc_info=True)
        raise



#--------------------------------------------------------------------------------------------------------------------------
#   Common Functions 
#--------------------------------------------------------------------------------------------------------------------------

    
    
## 참고용: logger 
## https://otzslayer.github.io/python/2021/11/26/logging-uncaught-exception-in-python.html
# def catch_exception(exc_type, exc_value, exc_traceback):
#     # Keyboard Interrupt를 통한 예외 발생은 무시합니다.
#     if issubclass(exc_type, KeyboardInterrupt):
#         sys.__excepthook__(exc_type, exc_value, exc_traceback)
#         return
#     error_logger.error(
#         "[TRACE-BACK]",
#         exc_info=(exc_type, exc_value, exc_traceback)
#         )
# # sys.excepthook을 대체합니다.
# sys.excepthook = catch_exception  