import os 
import logging
import logging.config
from datetime import datetime

#--------------------------------------------------------------------------------------------------------------------------
#    GLOBAL VARIABLE
#--------------------------------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------------------------------
#    ProcessLogger Class : (ALO master에서만 사용)
#--------------------------------------------------------------------------------------------------------------------------
# TODO https://betterstack.com/community/questions/how-to-color-python-logging-output/ 등 참고하여 log config의 stream handler에 색깔 넣는 방법 있다면 현재 print_color로 출력후 file handler만 쓴는 방식 수정하는게 좋을듯 
class ProcessLogger: 
    # [%(filename)s:%(lineno)d]
    # envs 미입력 시 설치 과정, 프로세스 진행 과정 등 전체 과정 보기 위한 로그 생성 
    def __init__(self, project_home: str):
        self.project_home = project_home
        self.train_log_path = self.project_home + ".train_artifacts/log/"
        self.inference_log_path = self.project_home + ".inference_artifacts/log/"
        if not os.path.exists(self.train_log_path):
            os.makedirs(self.train_log_path)
        if not os.path.exists(self.inference_log_path):
            os.makedirs(self.inference_log_path)
        # 현재 pipeline 등 환경 정보를 알기 전에 큼직한 단위로 install 과정 등에 대한 logging을 alo master에서 진행 가능하도록 config
        self.process_logging_config = { 
            "version": 1,
            "formatters": {
                "complex": {
                    "format": f"[%(asctime)s][PROCESS][%(levelname)s]: %(message)s"
                },
                "meta": {
                    "format": f"[%(asctime)s][PROCESS][%(levelname)s][META]: %(message)s"
                }
            },
            "handlers": {
                "file_train": {
                    "class": "logging.FileHandler",
                    "filename": self.train_log_path + "process.log", 
                    "formatter": "complex",
                    "level": "INFO",
                },
                "file_inference": {
                    "class": "logging.FileHandler",
                    "filename": self.inference_log_path + "process.log", 
                    "formatter": "complex",
                    "level": "INFO",
                },
            },
            "root": {"handlers": ["file_train", "file_inference"], "level": "INFO"},
            "loggers": {"ERROR": {"level": "ERROR"}, "WARNING": {"level": "WARNING"}, "INFO": {"level": "INFO"}}
        }
        self.meta_logging_config = { 
            "version": 1,
            "formatters": {
                "meta": {
                    "format": f"[%(asctime)s][PROCESS][META]: %(message)s"
                }
            },
            "handlers": {
                "file_train": {
                    "class": "logging.FileHandler",
                    "filename": self.train_log_path + "process.log", 
                    "formatter": "meta",
                    "level": "INFO",
                },
                "file_inference": {
                    "class": "logging.FileHandler",
                    "filename": self.inference_log_path + "process.log", 
                    "formatter": "meta",
                    "level": "INFO",
                },
            },
            "root": {"handlers": ["file_train", "file_inference"], "level": "INFO"},
            "loggers": {"INFO": {"level": "INFO"}}
        }
    #--------------------------------------------------------------------------------------------------------------------------
    #    Process Logging API
    #--------------------------------------------------------------------------------------------------------------------------
    # https://velog.io/@qlgks1/python-python-logging-%ED%95%B4%EB%B6%80
    def process_meta(self, msg, color='cyan'):
        # print
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        formatted_msg = f"[{time_utc}][PROCESS][META]: {msg}"
        # 어짜피 process_info는 내부 개발자만 쓸거기 때문에 raise ValueError해도 됨 
        self.print_color(formatted_msg, color)
        # file save 
        logging.config.dictConfig(self.meta_logging_config)
        # info logger을 meta logger로 상속 (참고: https://www.daleseo.com/python-logging-config/)
        meta_logger = logging.getLogger("INFO")
        meta_logger.info(f'{msg}')
        
        
    def process_info(self, msg, color='blue'):
        # print
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        formatted_msg = f"[{time_utc}][PROCESS][INFO]: {msg}"
        # 어짜피 process_info는 내부 개발자만 쓸거기 때문에 raise ValueError해도 됨 
        if color not in ['blue', 'green']:
            raise ValueError(f"[PROCESS][ERROR] only << blue >> or << green >> is allowed for process_info()")
        self.print_color(formatted_msg, color)
        # file save 
        logging.config.dictConfig(self.process_logging_config)
        info_logger = logging.getLogger("INFO") 
        info_logger.info(f'{msg}')


    def process_warning(self, msg):
        # print
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        formatted_msg = f"[{time_utc}][PROCESS][WARNING]: {msg}"
        self.print_color(formatted_msg, 'yellow')
        # file save 
        logging.config.dictConfig(self.process_logging_config)
        warning_logger = logging.getLogger("WARNING") 
        warning_logger.warning(f'{msg}')


    def process_error(self, msg):
        # print
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        formatted_msg = f"[{time_utc}][PROCESS][ERROR]: {msg}"
        
        # file save 
        logging.config.dictConfig(self.process_logging_config)
        error_logger = logging.getLogger("ERROR") 
        error_logger.error(f'{msg}')
        # FIXME train만 돌든 inference만 돌든 일단 artifacts 폴더는 둘다 만들기 때문에 process error도 둘 다에 백업 
        try:
            backup_error_artifacts(self.project_home)
        except: 
            raise NotImplementedError("Failed to backup artifacts before raising << process error >>")
        finally: 
            raise ValueError(formatted_msg)  

    
    def print_color(self, msg, _color):
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
            print(COLOR_DICT[_color.upper()] + msg+COLOR_END)
        else:
            raise ValueError('[ASSET][ERROR] print_color() function call error. - selected color : {}'.format(COLOR_DICT.keys()))

    
class Logger: 
    # [%(filename)s:%(lineno)d]
    # envs 미입력 시 설치 과정, 프로세스 진행 과정 등 전체 과정 보기 위한 로그 생성 
    def __init__(self, envs):
        self.asset_envs = envs
        self.project_home = self.asset_envs['project_home']
        self.pipeline = self.asset_envs['pipeline']
        self.step = self.asset_envs['step']
        try: 
            self.log_file_path = self.asset_envs['log_file_path']
        except: 
            raise ValueError('[ASSET][ERROR] Logger class argument << envs >> has no key of << log_file_path >>') 
        # (ref. : https://www.daleseo.com/python-logging-config/) / https://betterstack.com/community/guides/logging/python/python-logging-best-practices/
        # logging conifg (pipeline name check는 이미 master 쪽 src/alo.py에서 진행 완료)
        # user API 용 logging config 
        self.user_logging_config = { 
            "version": 1,
            "formatters": {
                "complex": {
                    "format": f"[%(asctime)s][USER][%(levelname)s][{self.pipeline}][{self.step}]: %(message)s"
                    #"datefmt": '%Y-%m-%d %H:%M:%S'
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "complex",
                    "level": "INFO",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": self.log_file_path, 
                    "formatter": "complex",
                    "level": "INFO",
                },
            },
            "root": {"handlers": ["console", "file"], "level": "INFO"},
            # error만 file only handling 
            "loggers": {"ERROR": {"level": "ERROR", "handlers": ["file"]}, "WARNING": {"level": "WARNING"}, "INFO": {"level": "INFO"}},
        }
        
        # ALO internally used logging config 
        # asset_log 시엔 print_color 및 별도 포맷팅을 위해 별도 config 구성  
        self.asset_logging_config = { 
            "version": 1,
            "formatters": {
                "complex": {
                    "format": f"[%(asctime)s][ASSET][%(levelname)s][{self.pipeline}][{self.step}]: %(message)s"
                    #"datefmt": '%Y-%m-%d %H:%M:%S'
                },
            },
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": self.log_file_path, 
                    "formatter": "complex",
                    "level": "INFO",
                },
            },
            "root": {"handlers": ["file"], "level": "INFO"},
            "loggers": {"ERROR": {"level": "ERROR"}, "WARNING": {"level": "WARNING"}, "INFO": {"level": "INFO"}},
        }
        
    #--------------------------------------------------------------------------------------------------------------------------
    #    User Logging API
    #--------------------------------------------------------------------------------------------------------------------------
    def user_info(self, msg):
        """ Description
            -----------
                - User Asset에서 알려주고 싶은 정보를 저장한다.
            Parameters
            -----------
                - msg (str) : 저장할 문자열 (max length : 255)
            Example
            -----------s
                - user_info('hello')
        """
        if not isinstance(msg, str):
            self.asset_error(f"Failed to user_info(). Only support << str >> type for the argument. \n You entered: {msg}")
        
        logging.config.dictConfig(self.user_logging_config)
        info_logger = logging.getLogger("INFO") 
        info_logger.info(f'{msg}')


    def user_warning(self, msg):
        """Description
            -----------
                - Asset에서 필요한 경고를 저장한다.
            Parameters
            -----------
                - msg (str) : 저장할 문자열 (max length : 255)
            Example
            -----------
                - user_warning('hello')
        """

        if not isinstance(msg, str):
            self.asset_error(f"Failed to user_warning(). Only support << str >> type for the argument. \n You entered: {msg}")

        logging.config.dictConfig(self.user_logging_config)
        warning_logger = logging.getLogger("WARNING") 
        warning_logger.warning(f'{msg}')


    # TODO save error 하면 마지막에 error를 pipeline.log 저장하고 죽는지 확인 필요 
    def user_error(self, msg):
        """Description
            -----------
                - Asset에서 필요한 에러를 저장한다.
            Parameters
            -----------
                - msg (str) : 저장할 문자열 (max length : 255)
            Example
            -----------
                - user_error('hello')
        """

        if not isinstance(msg, str):
            self.asset_error(f"Failed to user_error(). Only support << str >> type for the argument. \n You entered: {msg}")
            
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        formatted_msg = "".join([
            f"\n\n============================= USER ERROR =============================\n",
            f"TIME(UTC)   : {time_utc}\n",
            f"PIPELINE    : {self.pipeline}\n",
            f"STEP        : {self.step}\n",
            f"ERROR(msg)  : {msg}\n",
            f"=======================================================================\n\n"])
        logging.config.dictConfig(self.user_logging_config)
        error_logger = logging.getLogger("ERROR") 
        error_logger.error(f'{formatted_msg}')

        try:
            backup_error_artifacts(self.project_home)
        except: 
            raise NotImplementedError("Failed to backup artifacts before raising << process error >>")
        finally: 
            raise ValueError(formatted_msg)  

    
    #--------------------------------------------------------------------------------------------------------------------------
    #    ALO Internal Logging
    #--------------------------------------------------------------------------------------------------------------------------
    # internal function 이므로 msg가 문자열인지 체크 X 
    def asset_info(self, msg, color='blue'): # color는 blue, green만 제공 
        # stdout 
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        formatted_msg = f"[{time_utc}][ASSET][INFO][{self.pipeline}][{self.step}]: {msg}"
        if color not in ['blue', 'green']:
            self.asset_error(f"[ASSET][ERROR] only << blue >> or << green >> is allowed for asset_info()")
        self.print_color(formatted_msg, color)
        # log file save 
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        error_logger = logging.getLogger("INFO") 
        error_logger.info(f'{msg}')

   
    def asset_warning(self, msg):
        # stdout
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        formatted_msg = f"[{time_utc}][ASSET][WARNING][{self.pipeline}][{self.step}]: {msg}"
        self.print_color(formatted_msg, 'yellow')
        # log file save 
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        error_logger = logging.getLogger("WARNING") 
        error_logger.warning(f'{msg}')
    
    
    def asset_error(self, msg):
        # stdout 
        time_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        # from pytz import timezone 
        # time_kst = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
        formatted_msg = "".join([
            f"\n\n============================= ASSET ERROR =============================\n",
            f"TIME(UTC)   : {time_utc}\n",
            f"PIPELINE    : {self.pipeline}\n",
            f"STEP        : {self.step}\n",
            f"ERROR(msg)  : {msg}\n",
            f"=======================================================================\n\n"])
        # log file save 
        logging.config.dictConfig(self.asset_logging_config) # file handler only logging config 
        error_logger = logging.getLogger("ERROR") 
        error_logger.error(f'{formatted_msg}') #, exc_info=True) #, stack_info=True, exc_info=True)
        try:
            backup_error_artifacts(self.project_home)
        except: 
            raise NotImplementedError("Failed to backup artifacts before raising << process error >>")
        finally: 
            raise ValueError(formatted_msg)  

    
    def print_color(self, msg, _color):
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
            print(COLOR_DICT[_color.upper()] + msg+COLOR_END)
        else:
            raise ValueError('[ASSET][ERROR] print_color() function call error. - selected color : {}'.format(COLOR_DICT.keys()))


#--------------------------------------------------------------------------------------------------------------------------
#   Common Functions 
#--------------------------------------------------------------------------------------------------------------------------

def backup_error_artifacts(project_home):
    """ Description
        -----------
            - 파이프라인 실행 종료 후 사용한 yaml과 결과 artifacts를 .history에 백업함 
        Parameters
        -----------
            - project_home: project home absolute path ~ str

        Return
        -----------
            - 'OK'
        Example
        -----------
            - backup_artifacts('/~/~/alo/')
    """
    PROJECT_HOME = project_home

    try:
        error_occur_time = datetime.now().strftime("%y%m%d_%H%M%S")
    
        # ALO에서 일반적으로 잘 수행된 backup artifacts 폴더 명은 프로세스 시작시간으로 시작하지만, error 발생 시엔 error_에러발생시간_~ 으로 폴더명 지정 
        backup_folder= f'{error_occur_time}_artifacts_error/'
        # TODO current_pipelines 는 차후에 workflow name으로 변경이 필요
        temp_backup_artifacts_dir = PROJECT_HOME + backup_folder
        
        # 임시 저장 폴더 만들기
        if os.path.exists(temp_backup_artifacts_dir):
            shutil.rmtree(temp_backup_artifacts_dir) 
        os.mkdir(temp_backup_artifacts_dir)
        
        # FIXME 어떤 plan yaml 파일을 썼는지 받아오기 어려우므로 config 폴더 통 째로 임시 폴더로 copy 하기 
        shutil.copytree(PROJECT_HOME + 'config', temp_backup_artifacts_dir + 'config')
        # 임시 폴더에 artifacts 들을 백업 (train, inference 상관없이)
        shutil.copytree(PROJECT_HOME + ".train_artifacts", temp_backup_artifacts_dir + ".train_artifacts")
        shutil.copytree(PROJECT_HOME + ".inference_artifacts", temp_backup_artifacts_dir + ".inference_artifacts")
            
        # 임시 폴더를 .history 밑으로 이동 
        shutil.move(temp_backup_artifacts_dir, PROJECT_HOME + ".history/")
        # 잘 move 됐는 지 확인  
        if os.path.exists(PROJECT_HOME + ".history/" + backup_folder):
            return 'OK'
    except:
        raise NotImplementedError("Failed to backup error artifacts.")
    finally:
        if os.path.exists(temp_backup_artifacts_dir):
            shutil.rmtree(temp_backup_artifacts_dir) # copy 실패 시 임시 backup_artifacts_home 폴더 삭제 