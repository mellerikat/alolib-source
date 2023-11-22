# -*- coding: utf-8 -*-

import os
import pkg_resources 
from datetime import datetime
from pytz import timezone
# FIXME # DeprecationWarning: pkg_resources is deprecated as an API. 해결 필요? 
import yaml
import pickle
import json 
import shutil
from alolib.logger import Logger 

#--------------------------------------------------------------------------------------------------------------------------
#    GLOBAL VARIABLE
#--------------------------------------------------------------------------------------------------------------------------
## inference output format 
# FIXME 대문자 일단 비허용 
CSV_FORMATS = {"*.csv"}
IMAGE_FORMATS = {"*.jpg", "*.jpeg", "*.png"}
#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------

class Asset:
    # def __init__(self, envs, argv, version='None'):
    def __init__(self, asset_structure):
        self.asset = self
        self.artifacts_structure = {
            'input': {}, 
            '.train_artifacts': {
                'score': {},
                'output': {},
                'log': {},
                'report': {},
                'models': {}
            },
            '.inference_artifacts': {
                'score': {},
                'output': {},
                'log': {},
            },
            '.asset_interface': {},
            '.history': {}
        }
        # FIXME 추후 alo 이름 변경 필요? 
        self.alolib_version = pkg_resources.get_distribution('alolib').version  
        # 1. set envs, args, data, config .. 
        try:
            self.asset_envs = asset_structure.envs
            self.alo_version = self.asset_envs['alo_version']
            self.asset_branch = self.asset_envs['asset_branch']
            self.solution_metadata_version = self.asset_envs['solution_metadata_version']
            self.save_artifacts_path = self.asset_envs['save_train_artifacts_path'] if self.asset_envs['pipeline'] == 'train_pipeline' else self.asset_envs['save_inference_artifacts_path']
            self.proc_start_time = self.asset_envs['proc_start_time'] # alo runs start time 
            # API 호출 했는 지 count 
            for k in ['load_data', 'load_config', 'save_data', 'save_config']:
                self.asset_envs[k] = 0 
            # 현재는 PROJECT PATH 보다 한 층 위 folder에서 실행 중 
            self.project_home = self.asset_envs['project_home']
            # log file path 
            self.artifact_dir = '.train_artifacts/' if self.asset_envs['pipeline'] == 'train_pipeline' else '.inference_artifacts/'
            self.log_file_path = self.project_home + self.artifact_dir + "log/pipeline.log"
            self.asset_envs['log_file_path'] = self.log_file_path
            
            self.asset_args = asset_structure.args
            self.asset_data = asset_structure.data
            self.asset_config = asset_structure.config
            
        except Exception as e:
            raise ValueError(e)
        
        # init logger 
        self.logger = Logger(self.asset_envs)
        
        # input 데이터 경로 
        self.input_data_home = self.project_home + "input/"
        # asset 코드들의 위치
        self.asset_home = self.project_home + "assets/"



        
    ##################################################################################################################################################################
    #                                                                           Slave API
    ##################################################################################################################################################################
    def save_info(self, msg):
        self.logger.user_info(msg)
        
    def save_warning(self, msg):
        self.logger.user_warning(msg)
        
    def save_error(self, msg):
        self.logger.user_error(msg)
  
    
    def load_args(self):
        """ Description
            -----------
                - Asset 에 필요한 args를 반환한다.
            Parameters
            -----------
            Return
            -----------
                - args  (dict)
            Example
            -----------
                - args = load_args()
        """
        return self.asset_args.copy()

    
    def load_config(self):
        """ Description
            -----------
                - Asset 에 필요한 config를 반환한다.
            Parameters
            -----------
            Return
            -----------
                - config  (dict)
            Example
            -----------
                - config = load_config()
        """
        if self.asset_envs['interface_mode'] == 'memory':
            self.asset_envs['load_config'] += 1
            return self.asset_config
        elif self.asset_envs['interface_mode'] == 'file':
            # FIXME 첫번째 step일 경우는 pkl 파일 저장된 거 없으므로 memory interface랑 동일하게 일단 진행 
            if self.asset_envs['num_step'] == 0: 
                self.asset_envs['load_config'] += 1
                return self.asset_config
            try:
                # 이전 스탭에서 저장했던 pkl을 가져옴 
                file_path = self.asset_envs['artifacts']['.asset_interface'] + self.asset_envs['pipeline'] + "/" + self.asset_envs['prev_step'] + "_config.pkl"
                config = load_file(file_path)
                self.asset_envs['load_config'] += 1
                return config
            except Exception as e:
                self.logger.asset_error(str(e))     
        else: 
            self.logger.asset_error(f"Only << file >> or << memory >> is supported for << interface_mode >>")
            
    # TODO 파일 모드 시 .asset_interface에 저장할 때 step별로 subdirectory로 나눌필요 추후 있을듯 
    def save_config(self, config):
        if not isinstance(config, dict):
            self.logger.asset_error("Failed to save_config(). only << dict >> type is supported for the function argument.")
        # asset_config update ==> decorator_run에서 다음 step으로 넘겨주는데 사용됨
        if self.asset_envs['interface_mode'] == 'memory':
            self.asset_envs['save_config'] += 1
            self.asset_config = config 
        elif self.asset_envs['interface_mode'] == 'file':
            try:
                dir_path = self.asset_envs['artifacts']['.asset_interface'] + self.asset_envs['pipeline'] + "/"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    self.logger.asset_info(f"<< {dir_path} >> directory created for << save_config >>")
                config_file = dir_path + self.asset_envs['step']
                if type(config) == dict:
                    config_file = config_file + "_config.pkl"
                else:
                    self.logger.asset_error("The type of data must be << dict >>")
                save_file(config, config_file)
                self.asset_config = config 
                self.asset_envs['save_config'] += 1
            except Exception as e:
                self.logger.asset_error(str(e))   
        else: 
            self.logger.asset_error(f"Only << file >> or << memory >> is supported for << interface_mode >>")  
        
    # FIXME data도 copy()로 돌려줄 필요 있을 지? 
    def load_data(self):
        """ Description
            -----------
                - Asset 에 필요한 data를 반환한다.
            Parameters
            -----------
            Return
            -----------
                - data  (dict)
            Example
            -----------
                - data = load_data()
        """
        if self.asset_envs['interface_mode'] == 'memory':
            self.asset_envs['load_data'] += 1
            return self.asset_data
        elif self.asset_envs['interface_mode'] == 'file':
            # FIXME 첫번째 step일 경우는 pkl 파일 저장된 거 없으므로 memory interface랑 동일하게 일단 진행 
            if self.asset_envs['num_step'] == 0: 
                self.asset_envs['load_data'] += 1
                return self.asset_data
            try:
                # 이전 스탭에서 저장했던 pkl을 가져옴 
                file_path = self.asset_envs['artifacts']['.asset_interface'] + self.asset_envs['pipeline'] + "/" + self.asset_envs['prev_step'] + "_data.pkl"
                data = load_file(file_path)
                self.asset_envs['load_data'] += 1
                return data
            except Exception as e:
                self.logger.asset_error(str(e))     
        else: 
            self.logger.asset_error(f"Only << file >> or << memory >> is supported for << interface_mode >>")
    
    # TODO 파일 모드 시 .asset_interface에 저장할 때 step별로 subdirectory로 나눌필요 추후 있을듯 
    def save_data(self, data):
        if not isinstance(data, dict):
            self.logger.asset_error("Failed to save_data(). only << dict >> type is supported for the function argument.")
        # asset_data update ==> decorator_run에서 다음 step으로 넘겨주는데 사용됨
        if self.asset_envs['interface_mode'] == 'memory':
            self.asset_envs['save_data'] += 1 # 이번 asset에서 save_data API를 1회만 잘 호출했는지 체크 
            self.asset_data = data
        elif self.asset_envs['interface_mode'] == 'file':
            try:
                dir_path = self.asset_envs['artifacts']['.asset_interface'] + self.asset_envs['pipeline'] + "/"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    self.logger.asset_info(f"<< {dir_path} >> directory created for << save_data >>")
                data_file = dir_path + self.asset_envs['step']
                if type(data) == dict:
                    data_file = data_file + "_data.pkl"
                else:
                    self.logger.asset_error("The type of data must be << dict >>")
                save_file(data, data_file)
                self.asset_data = data
                self.asset_envs['save_data'] += 1
            except Exception as e:
                self.logger.asset_error(str(e))   
        else: 
            self.logger.asset_error(f"Only << file >> or << memory >> is supported for << interface_mode >>")  
        
    def load_summary(self):
        yaml_dict = dict()
        yaml_file_path = None 
        
        # self.asset_envs['pipeline'] 는 main.py에서 설정 
        if self.asset_envs['pipeline']  == "train_pipeline":
            yaml_file_path = self.asset_envs["artifacts"][".train_artifacts"] + "score/" + "train_summary.yaml" 
        elif self.asset_envs['pipeline'] == "inference_pipeline":
            yaml_file_path = self.asset_envs["artifacts"][".inference_artifacts"] + "score/" + "inference_summary.yaml" 
        else: 
            self.logger.asset_error(f"You have written wrong value for << asset_source  >> in the config yaml file. - { self.asset_envs['pipeline']} \n Only << train_pipeline >> and << inference_pipeline >> are permitted")

        # 이전의 asset 들 중에서 save_summary를 이미 한 상태여야 summary yaml 파일이 존재할 것이고, load가 가능함 
        if os.path.exists(yaml_file_path):
            try:
                with open(yaml_file_path, encoding='UTF-8') as f:
                    yaml_dict  = yaml.load(f, Loader=yaml.FullLoader)
            except:
                self.logger.asset_error(f"Failed to call << load_summary>>. \n - summary yaml file path : {yaml_file_path}")
        else: 
            self.logger.asset_error(f"Failed to call << load_summary>>. \n You did not call << save_summary >> in previous assets before calling << load_summary >>")
        
        return yaml_dict 
    
    
    # FIXME 사실 save summary 는 inference pipeline에서만 실행하겠지만, 현 코드는 train에서도 되긴하는 구조 
    def save_summary(self, result, score, note="", probability={}):
        
        """ Description
            -----------
                - train_summary.yaml (train 시에도 학습을 진행할 시) 혹은 inference_summary.yaml을 저장합니다. 
                - 참고 CLM: http://collab.lge.com/main/display/LGEPROD/AI+Advisor+Data+Flow#Architecture--1257299622
                - 숫자 형은 모두 소수점 둘째 자리까지 표시합니다.  
            Parameters
            -----------
                - result: Inference result summarized info. (str, length limit: 25) 
                - score: model performance score to be used for model retraining (float, 0 ~ 1.0)
                - note: optional & additional info. for inference result (str, length limit: 100) (optional)
                - probability: Classification Solution의 경우 라벨 별로 확률 값을 제공합니다. (dict - key:str, value:float) (optional)
                            >> (ex) {'OK': 0.6, 'NG':0.4}
            Return
            -----------
                - summaray_data: summary yaml 파일에 저장될 데이터 (dict) 
            Example
            -----------
                - summary_data = save_summary(result='OK', score=0.613, note='aloalo.csv', probability={'OK':0.715, 'NG':0.135, 'NG1':0.15})
        """
        result_len_limit = 32
        note_len_limit = 128 
        
        # result는 문자열 12자 이내인지 확인
        if not isinstance(result, str) or len(result) > result_len_limit:
            self.logger.asset_error(f"The length of string argument << result >>  must be within {result_len_limit} ")
        
        # score는 0 ~ 1.0 사이의 값인지 확인
        if not isinstance(score, (int, float)) or not 0 <= score <= 1.0:
            self.logger.asset_error("The value of float (or int) argument << score >> must be between 0.0 and 1.0 ")

        # note는 문자열 100자 이내인지 확인
        if not isinstance(result, str) or len(note) > note_len_limit:
            self.logger.asset_error(f"The length of string argument << note >>  must be within {note_len_limit} ")
                    
        # probability가 존재하는 경우 dict인지 확인
        if (probability is not None) and (not isinstance(probability, dict)):
            self.logger.asset_error("The type of argument << probability >> must be << dict >>")
            
        # probability key는 string이고 value는 float or int인지 확인 
        key_chk_str_set = set([isinstance(k, str) for k in probability.keys()])
        value_type_set = set([type(v) for v in probability.values()])
        if key_chk_str_set != {True}: 
            self.logger.asset_error("The key of dict argument << probability >> must have the type of << str >> ")
        if not value_type_set.issubset({float, int}): 
            self.logger.asset_error("The value of dict argument << probability >> must have the type of << int >> or << float >> ")
        # probability value 합이 1인지 확인 
        if sum(probability.values()) != 1: 
            self.logger.asset_error("The sum of probability dict values must be << 1.0 >>")
        
        # FIXME 가령 0.50001, 0.49999 같은건 대응이 안됨 
        # FIXME 처음에 사용자가 입력한 dict가 합 1인지도 체크필요 > 부동소수 에러 예상
        def make_addup_1(prob): #inner func. / probability 합산이 1이되도록 가공, 소수 둘째 자리까지 표시 
            max_value_key = max(prob, key=prob.get) 
            proc_prob_dict = dict()  
            for k, v in prob.items(): 
                if k == max_value_key: 
                    proc_prob_dict[k] = 0 
                    continue
                proc_prob_dict[k] = round(v, 2) # 소수 둘째자리
            proc_prob_dict[max_value_key] = round(1 - sum(proc_prob_dict.values()), 2)
            return proc_prob_dict
           
        # FIXME .inference_artifacts/output/[현재 step >> 대부분 inference일 것] 내에 output 파일이 없으면 에러         
        output_file_path = self.artifact_dir + 'output/' + self.asset_envs['step']
        if len(os.listdir(output_file_path))==0:
            self.logger.asset_error("Failed to save summary. Please generate inference output files first. \n (ex. output.csv, output.jpg)")
        
        # .inference_artifacts/output 내의 파일의 확장자가 지원하지 않는 타입이면 에러 
        for output_file in os.listdir(output_file_path):
            _, extension = os.path.splitext(output_file)
            # 확장자 대문자로 입력했으면 에러 
            if extension.isupper() == True: 
                self.logger.asset_error(f"Please save the inference output file extension in lowercase letters. \n You entered: {output_file}")
            # 확장자가 지원하지 않는 타입이면 에러 
            if '*' + extension not in CSV_FORMATS.union(IMAGE_FORMATS): 
                self.logger.asset_error(f"Unsupported type of extension: {output_file} \n >> Available extensions: {CSV_FORMATS.union(IMAGE_FORMATS)} \n (ex. output.csv, output.jpg)")

        # file_path 생성
        file_path = ""     
        # external save artifacts path 
        if self.save_artifacts_path is None: 
            mode = self.asset_envs['pipeline'].split('_')[0] # train or inference
            self.logger.asset_warning(f"Please enter the << external_path - save_{mode}_artifacts_path >> in the experimental_plan.yaml.")
        else: 
            file_path = self.save_artifacts_path 
        
        # version은 str type으로 포맷팅 
        ver = ""
        if self.solution_metadata_version == None: 
            self.solution_metadata_version = ""
        else: 
            ver = 'v' + str(self.solution_metadata_version)
        
        # FIXME 배포 테스트 시 probability의 key 값 (클래스)도 정확히 모든 값 기입 됐는지 체크 필요     
        # dict type data to be saved in summary yaml 
        summary_data = {
            'result': result,
            'score': round(score, 2), # 소수 둘째자리
            'date':  datetime.now(timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S'), 
            # FIXME note에 input file 명 전달할 방법 고안 필요 
            'note': note,
            'probability': make_addup_1(probability),
            'file_path': file_path,  # external save artifacts path
            'version': ver
        }

        # self.asset_envs['pipeline'] 는 main.py에서 설정 
        if self.asset_envs['pipeline']  == "train_pipeline":
            file_path = self.asset_envs["artifacts"][".train_artifacts"] + "score/" + "train_summary.yaml" 
        elif self.asset_envs['pipeline'] == "inference_pipeline":
            file_path = self.asset_envs["artifacts"][".inference_artifacts"] + "score/" + "inference_summary.yaml" 
        else: 
            self.logger.asset_error(f"You have written wrong value for << asset_source  >> in the config yaml file. - { self.asset_envs['pipeline']} \n Only << train_pipeline >> and << inference_pipeline >> are permitted")
        
        # save summary yaml 
        try:      
            with open(file_path, 'w') as file:
                yaml.dump(summary_data, file, default_flow_style=False)
            self.logger.asset_info(f"Successfully saved inference summary yaml. \n >> {file_path}", color='green')
        except: 
            self.logger.asset_error(f"Failed to save summary yaml file \n @ << {file_path} >>")
             
        return summary_data

    # FIXME 만약 inference pipeline 여러개인 경우 model 파일이름을 사용자가 잘 분리해서 사용해야함 > pipline name 인자 관련 생각필요 
    # FIXME multi train, inference pipeline 일 때 pipeline name (yaml에 추가될 예정)으로 subfloder 분리해서 저장해야한다. (step이름 중복 가능성 존재하므로)
    # >> os.envrion 변수에 저장하면 되므로 사용자 파라미터 추가할 필욘 없을 듯
    # FIXME  step명은 같은걸로 pair, train-inference만 예외 pair / 단, cascaded train같은경우는 train1-inference1, train2-inference2가 pair  식으로
    def get_model_path(self, use_inference_path=False): # get_model_path 는 inference 시에 train artifacts 접근할 경우도 있으므로 pipeline_mode랑 step_name 인자로 받아야함 
        """ Description
            -----------
                - model save 혹은 load 시 필요한 model path를 반환한다. 
            Parameters
            -----------
                - use_inference_path: default는 False여서, train pipeline이던 inference pipline이던 train model path 반환하고,
                                      예외적으로 True로 설정하면 inference pipeline이 자기 자신 pipeline name의 subfolder를 반환한다. (bool)
            Return
            -----------
                - model_path: model 경로 
            Example
            -----------
                - model_path = get_model_path(use_inference_path=False)
        """
        # use_inference_path type check 
        if not isinstance(use_inference_path, bool):
            self.logger.asset_error("The type of argument << use_inference_path >>  must be << boolean >> ")
        
        # yaml에 잘 "train_pipeline" or "inference_pipeline" 라고 잘 입력했는지 체크
        allowed_pipeline_mode_list = ["train_pipeline",  "inference_pipeline"]
        current_pipe_mode = self.asset_envs['pipeline']
        if current_pipe_mode not in allowed_pipeline_mode_list: 
            self.logger.asset_error(f"You entered the wrong parameter for << user_parameters >> in your config yaml file : << {current_pipe_mode} >>. \n L ""You can select the pipeline_mode among << {allowed_pipeline_mode_list} >>"" ")
        
        # create step path 
        # default로는 step name은 train-inference 혹은 inference1-inference2 (추후 pipeline_name 추가 시) 간의 step 명이 같은 것 끼리 pair 
        # self.asset_envs['step']는 main.py에서 설정 
        current_step_name = self.asset_envs['step'] 

        # TODO train2도 train등으로 읽어 오게 수정 논의 필요
        current_step_name = ''.join(filter(lambda x: x.isalpha() or x == '_', current_step_name))

        # TODO use_inference_path true인 경우 inference path 사용하게 수정
        artifacts_name = ".train_artifacts"
        if use_inference_path == True and current_pipe_mode == "inference_pipeline":
            # infernce path를 사용
            artifacts_name = ".inference_artifacts"
        elif use_inference_path == True and current_pipe_mode != "inference_pipeline":
            self.logger.asset_error("If you set 'use_inference_path' to True, it should operate in the inference pipeline.")

        model_path = self.asset_envs["artifacts"][artifacts_name] + f"models/{current_step_name}/"

        # inference pipeline 때 train artifacts에 같은 step 이름 없으면 에러 
        if (current_pipe_mode  == "inference_pipeline") and (current_step_name != "inference"):
            if not os.path.exists(model_path):
                self.logger.asset_error(f"You must execute train pipeline first. There is no model path : \n {model_path}") 
        # FIXME pipeline name 관련해서 추가 yaml 인자 받아서 추가 개발 필요 
        elif (current_pipe_mode  == "inference_pipeline") and (current_step_name == "inference"):
            model_path = self.asset_envs["artifacts"][artifacts_name] + f"models/train/"
            if not os.path.exists(model_path): 
                self.logger.asset_error(f"You must execute train pipeline first. There is no model path : \n {model_path}") 
            elif (os.path.exists(model_path)) and (len(os.listdir(model_path)) == 0): 
                self.logger.asset_error(f"You must generate train model first. There is no model in the train model path : \n {model_path}") 
                    
        # trian 땐 없으면 폴더 생성 
        os.makedirs(model_path, exist_ok=True) # exist_ok =True : 이미 존재하면 그대로 둠 
        self.logger.asset_info(f"Successfully got model path for saving or loading your AI model: \n {model_path}", "green")
        
        return model_path

    # FIXME multi train, inference pipeline 일 때 pipeline name (yaml에 추가될 예정)으로 subfloder 분리해서 저장해야한다. 파일이름이 output.csv, output.jpg로 고정이므로 
    # >> os.envrion 변수에 저장하면 되므로 사용자 파라미터 추가할 필욘 없을 듯 
    def get_output_path(self):
        """ Description
            -----------
                - train 혹은 inference artifacts output을 저장할 경로 반환 
                - 이름은 output.csv, output.jpg 로 고정 (정책)
            Parameters
            -----------
            Return
            -----------
                - output_path: 산출물을 저장할 output 경로 
            Example
            -----------
                - output_path = get_output_path("csv")
        """
        # yaml에 잘 "train_pipeline" or "inference_pipeline" 라고 잘 입력했는지 체크
        # self.asset_envs['pipeline'] check 
        # self.asset_envs['pipeline']은 main.py에서 설정
        allowed_pipeline_mode_list = ["train_pipeline",  "inference_pipeline"]
        current_pipe_mode = self.asset_envs['pipeline']
        if current_pipe_mode  not in allowed_pipeline_mode_list: 
            self.logger.asset_error(f"You entered the wrong parameter for << user_parameters >> in your config yaml file : << {current_pipe_mode} >>. \n - ""You can select the pipeline_mode among << {allowed_pipeline_mode_list} >>"" ")
        
        # create output path 
        output_path = ""
        current_step_name = self.asset_envs['step'] 
        if  current_pipe_mode == "train_pipeline":
            output_path = self.asset_envs["artifacts"][".train_artifacts"] + f"output/{current_step_name}/"
            os.makedirs(output_path, exist_ok=True) # exist_ok =True : 이미 존재하면 그대로 둠 
        elif current_pipe_mode == 'inference_pipeline': 
            output_path = self.asset_envs["artifacts"][".inference_artifacts"] + f"output/{current_step_name}/"
            os.makedirs(output_path, exist_ok=True)
            
        self.logger.asset_info(f"Successfully got << output path >> for saving your data into csv or jpg file: \n {output_path} \n - [NOTE] ""The names of output file must be fixed as << output.csv, output.jpg >>"" ", "green")
        
        return output_path

    def get_report_path(self):
        """ Description
            -----------
                - report 를 저장할 path 반환. report는 train pipeline에서만 생성 (정책)
                
            Parameters
            -----------

            Return
            -----------
                - report_path: report.html을 저장할 output 경로 
            Example
            -----------
                - report_path = get_report_path()
        """
        
        # self.asset_envs['pipeline'] check >> train pipeline만 허용! 
       # self.asset_envs['pipeline']은 main.py에서 설정
        allowed_pipeline_mode_list = ["train_pipeline"]
        current_pipe_mode = self.asset_envs['pipeline']
        if current_pipe_mode  not in allowed_pipeline_mode_list: 
            self.logger.asset_error(f"You entered the wrong parameter for << user_parameters >> in your config yaml file : << {current_pipe_mode} >>. \n L ""You can select the pipeline_mode among << {allowed_pipeline_mode_list} >>"" ")
            
        # create report path 
        report_path = self.asset_envs["artifacts"][".train_artifacts"] + "report/"
        os.makedirs(report_path, exist_ok=True) # exist_ok =True : 이미 존재하면 그대로 둠 

        report_path  = report_path
        self.logger.asset_info(f"Successfully got << report path >> for saving your << report.html >> file: \n {report_path}", "green")
        
        return report_path
    
    
    ##################################################################################################################################################################
    
    ##################################################################################################################################################################
    def _check_dataframe_key(self, prev_data):
        prev_keys = [i for i in prev_data.keys() if 'dataframe' in i]
        cur_keys =  [i for i in self.asset_data.keys() if 'dataframe' in i]
        if len(prev_keys) < len(cur_keys): 
            self.logger.asset_error(f"Do not add keys containing the word << dataframe >> into the output data dict to be saved.:\n You added: {set(cur_keys) - set(prev_keys)}")
        if len(prev_keys) > len(cur_keys): 
            self.logger.asset_error(f"Do not remove keys contaning the word << dataframe >>. \n You removed: {set(prev_keys) - set(cur_keys)}")
        if prev_keys != cur_keys: 
            self.logger.asset_error(f"Do not modify keys contaning the word << dataframe >>. \n - Previous step: {prev_keys} \n - Current step: {cur_keys}") 
        
    # TODO : check whether data & config are updated 필요할지? 
    # FIXME : 만약 config, data에 대해서 dict 타입으로 비교할 때 data dict 내에 dataframe 있으면 ValueError: Can only compare identically-labeled DataFrame objects 에러 발생 가능성 존재 
    def decorator_run(func):
        def _run(self, *args, **kwargs):
            step = self.asset_envs["step"]
            prev_data, prev_config = self.asset_data, self.asset_config 
            try:
                # print asset start info. 
                self._asset_start_info() 
                # run user asset 
                func(self, *args, **kwargs)
                # save_data, save_config 호출은 step 당 1회 제한 (안그러면 덮어씌워짐)
                if (self.asset_envs['save_data'] != 1) or (self.asset_envs['save_config'] != 1):
                        self.logger.asset_error(f"You did not call (or more than once) the \n << self.asset.save_data() >> \
                                            or << self.asset.save_conifg() >> API in the << {step} >> step. \n Both of calls are mandatory.")
                if (not isinstance(self.asset_data, dict)) or (not isinstance(self.asset_config, dict)):
                    self.logger.asset_error(f"You should make dict for argument of << self.asset.save_data()>> or << self.asset.save_config() >> \n @ << {step} >> step.")  
                # input step 이외에, 이번 step에서 사용자가 dataframe이라는 문자를 포함한 key를 새로 추가하지 않았는 지 체크 
                if step != 'input':
                    self._check_dataframe_key(prev_data)
            except:
                self.logger.asset_error(f"Failed to run << {step} >>")
                
            # print asset finish info.
            self._asset_finish_info()
            
            return self.asset_data, self.asset_config  
        
        return _run


    def check_args(self, arg_key, is_required=False, default="", chng_type="str" ):
        """ Description
            -----------
                Check user parameter. Replace value & type 

            Parameters
            -----------
                arg_key (str) : 사용자 라미미터 이름 
                is_required (bool) : 필수 존재 여부 
                default (str) : 사용자 파라미터가 존재하지 않을 경우, 강제로 입력될 값
                chng_type (str): 타입 변경 list, str, int, float, bool, 

            Return
            -----------
                arg_value (the replaced string)

            Example
            -----------
                x_columns  = self.asset.check_args(arg_key="x_columns", is_required=True, chng_type="list")
        """
        if is_required:
            try:
                arg_value = self.asset_args[arg_key] if self.asset_args[arg_key] is not None else ""
            except:
                self.logger.asset_error('>> Not found args [{}]'.format(arg_key))
        else:
            try:
                if type(self.asset_args[arg_key]) == type(None):
                    arg_value = default
                else:
                    arg_value = self.asset_args[arg_key] if self.asset_args[arg_key] is not None else ""
            except:
                arg_value = default
                
        chk_type = type(arg_value) ## issue: TypeError: 'str' object is not callable
        if chk_type == list:
            pass
        else:
            arg_value = self._convert_variable_type(arg_value, chng_type)

        return arg_value

        
# --------------------------------------------------------------------------------------------------------------------------
#    COMMON FUNCTION
# --------------------------------------------------------------------------------------------------------------------------

        
    def _convert_variable_type(self, variable, target_type):
        if not isinstance(target_type, str) or target_type.lower() not in ["str", "int", "float", "list", "bool"]:
            raise ValueError("Invalid target_type. Allowed values are 'str', 'int', 'float', and 'list'.")

        if target_type.lower() == "str" and not isinstance(variable, str):
            return str(variable)
        elif target_type.lower() == "int" and not isinstance(variable, int):
            return int(variable)
        elif target_type.lower() == "float" and not isinstance(variable, float):
            return float(variable)
        elif target_type.lower() == "list" and not isinstance(variable, list):
            return [variable]
        elif target_type.lower() == "bool" and not isinstance(variable, bool):
            if variable == "false" or variable == "False":
                return False
            else:
                return True
        else:
            return variable

                
    def _asset_start_info(self):
        msg = "".join([
            f"\n\n============================= ASSET START =============================\n",
            f"- time (UTC)        : {datetime.now(timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"- current step      : {self.asset_envs['step']}\n",
            f"- asset branch.     : {self.asset_branch}\n", 
            f"- alolib ver.       : {self.alolib_version}\n",
            f"- alo ver.          : {self.alo_version}\n",
            f"- load envs. keys   : {self.asset_envs.keys()}\n",
            f"- load args. keys   : {self.asset_args.keys()}\n",
            f"- load config. keys : {self.asset_config.keys()}\n", 
            f"- load data keys    : {self.asset_data.keys()}\n",
            f"=======================================================================\n\n"])
        self.logger.asset_info(msg)


    def _asset_finish_info(self): 
        msg = "".join([
            f"\n\n============================= ASSET FINISH ===========================\n",
            f"- time (UTC)        : {datetime.now(timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"- current step      : {self.asset_envs['step']}\n",
            f"- save config. keys : {self.asset_config.keys()}\n",
            f"- save data keys    : {self.asset_data.keys()}\n",
            f"=======================================================================\n\n"])
        self.logger.asset_info(msg)


#--------------------------------------------------------------------------------------------------------------------------
#   Common Functions 
#--------------------------------------------------------------------------------------------------------------------------


# FIXME load_file 함수 print, error 함수 변경필요 
def load_file(_data_file, _print=True):
    """ Description
        -----------
            - 파일을 로하하여 데이터로 가져온다.
        Parameters
        -----------
            - data_file (str) : 로드할 데이터의 파일이름 (경로 포함)
                                (확장자 지원 : csv, h5, tfrecord, pkl, json, params, log)
            - option
                - _print (bool) : 데이터 저장여부 출력
        Return
        -----------
            - data (csv) : dataframe
                    (pkl, json, params, log) : dictionary
        Example
        -----------
            - data = load_file(data_file)
    """

    _data = None
    if _data_file != None:
        try:
            if _data_file.lower().endswith('.pkl'):
                with open(_data_file, 'rb') as f:
                    _data = pickle.load(f)
            elif _data_file.lower().endswith('.json') or \
                    _data_file.lower().endswith('.params') or \
                    _data_file.lower().endswith('.log'):
                with open(_data_file, 'r') as f:
                    _data = json.load(f)
            else:
                raise TypeError('No Support file format (support : pkl, json, params, log)')
                
            if _print == True:
                print('Loaded : {}'.format(_data_file))
            else:
                pass
        except FileNotFoundError: 
            raise ValueError('File Not Found : {}'.format(_data_file))
        except AttributeError:
            # ex. tfrecord를 제작할 때 사용한 pandas 버전과 다른 경우 발생
            raise ValueError('Attribute Error')
        except:
            raise ValueError('File Data Error : {}'.format(_data_file))
    else:
        raise ValueError('Failed to load data. Data file path is None.')

    return _data

# FIXME save_file 함수 print, error 함수 변경필요 
def save_file(_data, _data_file, _print=True):
    """ Description
        -----------
            - 데이터를 파일로 저장한다.
        Parameters
        -----------
            - data : 파일로 저장할 데이터 (dict) : dataframe 등 key 포함 

            - data_file (str) : 저장할 데이터의 파일이름 (경로 포함)
                                (확장자 지원 : csv, pkl, json, params, log)
            - option
                - _print (bool) : 데이터 저장여부 출력
        Example
        -----------
            - save_file(data, data_file)
    """

    if _data_file != None and not (isinstance(_data, str) and _data == 'none') and len(_data) > 0:
        try:
            check_path(_data_file)

            if _data_file.lower().endswith('.pkl'):
                with open(_data_file, "wb") as f:
                    pickle.dump(_data, f)
            elif _data_file.lower().endswith('.json') or \
                    _data_file.lower().endswith('.params') or \
                    _data_file.lower().endswith('.log'):
                with open(_data_file, "w") as f:
                    # ensure_ascii=False : 한글 지원
                    json.dump(_data, f, indent=4, ensure_ascii=False)
            else:
                raise TypeError('No Support file format (support : pkl, json, params, log)')
            
            if _print == True:
                _msg = f'Saved : {_data_file}'
                print(_msg)
            else:
                pass
        except TypeError as e:
            raise TypeError(str(e))
        
        except:
            raise ValueError('Failed to save : {}'.format(_data_file))

    else:
        pass


def check_path(_filename, remake=False):
    """ Description
        -----------
            Check the directory of file

        Parameters
        -----------
            _filename (str) : the file name with full directory 
            options
                remake (bool) : Create the path if not path

        Return
        -----------
            bool

        Example
        -----------
            check_path('/home/user/work/filename.csv', remake=True)
    """
 
    # 접근 폴더가 없다면
    if not os.path.exists(os.path.dirname(_filename)):
        os.makedirs(os.path.dirname(_filename))
        return False
    else:
        # 폴더가 있다면 삭제하고, 다시 만든다.
        if remake == True:
            shutil.rmtree(os.path.dirname(_filename))
            os.makedirs(os.path.dirname(_filename))
        return True