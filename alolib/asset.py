# -*- coding: utf-8 -*-
import alolib 
import configparser 
from datetime import datetime
import os
import psutil
import yaml
from functools import wraps
from memory_profiler import profile
from pprint import pformat
from pytz import timezone
from alolib.logger import Logger 
from alolib.utils import display_resource, load_file, save_file, _convert_variable_type, _extract_partial_data

#--------------------------------------------------------------------------------------------------------------------------
#    GLOBAL VARIABLE
#--------------------------------------------------------------------------------------------------------------------------
## inference output format allowed
CSV_FORMATS = {"*.csv"}
IMAGE_FORMATS = {"*.jpg", "*.jpeg", "*.png"}
## {read_custom_config} format allowed  
CUSTOM_CONFIG_FORMATS = {".ini", ".yaml"}
#--------------------------------------------------------------------------------------------------------------------------

class Asset:
    def __init__(self, asset_structure):
        """ Initialize Asset class with {asset_structure}

        Args:
            asset_structure (object): asset info. structure (instance)

        Returns: -

        """
        self.asset = self
        ## alolib version (@ __init__.py)
        self.alolib_version = alolib.__version__ 
        ## set envs, args, data, config .. 
        ## asset envs
        self.asset_envs = asset_structure.envs
        ## set path and logger 
        self.project_home = self.asset_envs['project_home']
        ## log file path 
        self.artifact_dir = 'train_artifacts/' if self.asset_envs['pipeline'] == 'train_pipeline' else 'inference_artifacts/'
        self.log_file_path = self.project_home + self.artifact_dir + "log/pipeline.log"
        self.asset_envs['log_file_path'] = self.log_file_path
        ## init logger 
        self.logger = Logger(self.asset_envs, 'ALO')
        self.user_asset_logger = Logger(self.asset_envs, 'USER')
        try:
            ## envs related info.
            self.alo_version = self.asset_envs['alo_version']
            self.asset_branch = self.asset_envs['asset_branch']
            self.solution_metadata_version = self.asset_envs['solution_metadata_version']
            self.save_artifacts_path = self.asset_envs['save_train_artifacts_path'] if self.asset_envs['pipeline'] == 'train_pipeline' \
                                        else self.asset_envs['save_inference_artifacts_path']
            ## alo master runs start time 
            self.proc_start_time = self.asset_envs['proc_start_time'] 
            ## Count the number of times the user API is called in accordance with the allowed number of calls
            for k in ['load_data', 'load_config', 'save_data', 'save_config']:
                self.asset_envs[k] = 0 
            ## input data path
            self.input_data_home = self.project_home + "input/"
            ## asset code home path
            self.asset_home = self.project_home + "assets/"
            ## asset args
            self.asset_args = asset_structure.args
            ## asset data
            self.asset_data = asset_structure.data
            ## asset config 
            self.asset_config = asset_structure.config
        except Exception as e:
            self.logger.asset_error(str(e))
    
    #--------------------------------------------------------------------------------------------------------------------------
    #                                                         UserAsset API
    #--------------------------------------------------------------------------------------------------------------------------
    
    def save_info(self, msg, show=False):
        """ User logger info API. If show=True, 
            display together with the final summary table at the end of ALO execution.

        Args:
            msg     (str): log message
            show    (bool): whether to show in alo finish

        Returns: -

        """
        assert show in [True, False]
        if show == True:  
            self.user_asset_logger.asset_info(msg, show=True)
        else:
            self.user_asset_logger.asset_info(msg)
        
    def save_warning(self, msg):
        """ User logger warning API. 

        Args:
            msg     (str): log message

        Returns: -

        """
        self.user_asset_logger.asset_warning(msg)  
        
    def save_error(self, msg):
        """ User error warning API. 

        Args:
            msg     (str): log message

        Returns: -

        """
        self.user_asset_logger.asset_error(msg)
        
    def get_input_path(self): 
        """ Return input data path

        Args: -

        Returns: 
            input data path 

        """
        return self.input_data_home + self.asset_envs['pipeline'].split('_')[0] + '/'
    
    def load_args(self):
        """ Return args needed for user asset

        Args: -

        Returns: 
            asset args

        """
        return self.asset_args.copy()
    
    def load_config(self):
        """ Return config needed for user asset

        Args: -

        Returns: 
            asset config

        """
        if self.asset_envs['interface_mode'] == 'memory':
            self.asset_envs['load_config'] += 1
            return self.asset_config
        elif self.asset_envs['interface_mode'] == 'file':
            ## FIXME If it is the first step, proceed as with the memory interface since there are no saved pkl files.
            if self.asset_envs['num_step'] == 0: 
                self.asset_envs['load_config'] += 1
                return self.asset_config
            try:
                ## Retrieve the pickle file saved from the previous step.
                file_path = self.asset_envs['artifacts']['.asset_interface'] + self.asset_envs['pipeline'] + "/" + self.asset_envs['prev_step'] + "_config.pkl"
                config = load_file(file_path)
                self.logger.asset_message("Loaded : {}".format(file_path))
                self.asset_envs['load_config'] += 1
                return config
            except Exception as e:
                self.logger.asset_error(str(e))     
        else: 
            self.logger.asset_error(f"Only << file >> or << memory >> is supported for << interface_mode >>")
    
    def read_custom_config(self, config_file_path): 
        """ - Return the contents by directly reading the file located at \
              {HOME}/alo/input/{pipeline}/ followed by the {config_file_path}
            - Supported extension type: .ini and .yaml

        Args: 
            config_file_path (str): custom config file path 
                                    e.g. '2024-01-01/config/custom_config.ini'

        Returns: 
            configparser    (object): when .ini
            yaml info.      (dict): when .yaml

        """
        ## config file type check 
        if not isinstance(config_file_path, str): 
            self.logger.asset_error(f"Failed to << read_custom_config >>. \n - << config_file_path >> must have string type. \n - Your input config_file_path: << {config_file_path} >>.") 
        ## create absolute path of custom config 
        abs_config_file_path = self.input_data_home + self.asset_envs['pipeline'].split('_')[0] + '/' + config_file_path
        ## check file existence
        if not os.path.exists(abs_config_file_path): 
            self.logger.asset_error(f"Failed to << read_custom_config >>. \n - << {abs_config_file_path} >> does not exist.") 
        try:
            _, file_extension = os.path.splitext(config_file_path)
            ## check custom config format check (.ini / .yaml)
            if file_extension not in CUSTOM_CONFIG_FORMATS:
                self.logger.asset_error(f"Failed to << read_custom_config >>. \n - Allowed custom config file path extension: {CUSTOM_CONFIG_FORMATS} \n - Your input config_file_path: << {config_file_path} >>. ")
            ## return ConfigParser object or Yaml dict
            if file_extension == '.ini': 
                custom_config = configparser.ConfigParser() 
                custom_config.read(abs_config_file_path, encoding='utf-8') 
                return custom_config 
            elif file_extension == '.yaml': 
                yaml_dict = dict()
                with open(abs_config_file_path, encoding='UTF-8') as f:
                    yaml_dict  = yaml.load(f, Loader=yaml.FullLoader)
                return yaml_dict 
        except: 
            self.logger.asset_error(f"Failed to read custom config from << {abs_config_file_path} >>")  
        
    def save_config(self, config):
        """ Pass the updated config from a specific asset to the next one after a user has performed load_config()

        Args: 
            config  (dict): config tobe saved

        Returns: -

        """
        if not isinstance(config, dict):
            self.logger.asset_error("Failed to save_config(). only << dict >> type is supported for the function argument.")
        if self.asset_envs['interface_mode'] == 'memory':
            self.asset_envs['save_config'] += 1
            self.asset_config = config 
        elif self.asset_envs['interface_mode'] == 'file':
            try:
                dir_path = self.asset_envs['artifacts']['.asset_interface'] + self.asset_envs['pipeline'] + "/"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    self.logger.asset_message(f"<< {dir_path} >> directory created for << save_config >>")
                config_file = dir_path + self.asset_envs['step'] + "_config.pkl"
                ## save config file 
                save_file(config, config_file)
                self.logger.asset_message("Saved : {config_file}")
                self.asset_config = config 
                self.asset_envs['save_config'] += 1
            except Exception as e:
                self.logger.asset_error(str(e))   
        else: 
            self.logger.asset_error(f"Only << file >> or << memory >> is supported for << interface_mode >>")  
    
    def load_data(self, partial_load = ''):
        """ Return the data required for the asset.
            e.g. data = load_data()
                 data = load_data(partial_load = '240101/source/') 
            
        Args: 
            partial_load (str): Load only keys that partially contain the specified string

        Returns: -

        """
        ## check partial load
        if len(partial_load) > 0:  
            ## arg type check - str
            if not isinstance(partial_load, str):
                self.logger.asset_error(f"The type of << partial_load >> must be string.")
        ## partial extract from asset data
        data = _extract_partial_data(self.asset_data, partial_load) if len(partial_load) > 0 else self.asset_data
        if len(data) == 0: 
            self.logger.asset_error(f"Failed to partially load data. No key exists containing << {partial_load} >> in loaded data.") 
        ## return or save asset data 
        if self.asset_envs['interface_mode'] == 'memory':
            self.asset_envs['load_data'] += 1
            return data
        elif self.asset_envs['interface_mode'] == 'file':
            ## FIXME If it is the first step, proceed as with the memory interface since there are no saved pkl files.
            if self.asset_envs['num_step'] == 0: 
                self.asset_envs['load_data'] += 1
                return data 
            try:
                ## get pickle file which was saved in previous step 
                file_path = self.asset_envs['artifacts']['.asset_interface'] + self.asset_envs['pipeline'] + "/" + self.asset_envs['prev_step'] + "_data.pkl"
                data = load_file(file_path)
                self.logger.asset_message("Loaded : {}".format(file_path))
                ## partial extract from asset data
                data = _extract_partial_data(data, partial_load) if len(partial_load) > 0 else data
                self.asset_envs['load_data'] += 1
                return data
            except Exception as e:
                self.logger.asset_error(str(e))     
        else: 
            self.logger.asset_error(f"Only << file >> or << memory >> is supported for << interface_mode >>")
    
    def save_data(self, data):
        """ Pass the data updated by a user in a specific asset to the next asset after the user has called load_data().
            
        Args: 
            data (dict): data dict tobe saved

        Returns: -

        """
        if not isinstance(data, dict):
            self.logger.asset_error("Failed to save_data(). only << dict >> type is supported for the function argument.")
        if self.asset_envs['interface_mode'] == 'memory':
            ## Check if the save_data API was called correctly just once in this asset
            self.asset_envs['save_data'] += 1 
            self.asset_data = data
        elif self.asset_envs['interface_mode'] == 'file':
            try:
                dir_path = self.asset_envs['artifacts']['.asset_interface'] + self.asset_envs['pipeline'] + "/"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    self.logger.asset_message(f"<< {dir_path} >> directory created for << save_data >>")
                data_file = dir_path + self.asset_envs['step'] + "_data.pkl"
                ## save file 
                save_file(data, data_file)
                self.logger.asset_message("Saved : {data_file}")
                ## set class variable 
                self.asset_data = data
                self.asset_envs['save_data'] += 1
            except Exception as e:
                self.logger.asset_error(str(e))   
        else: 
            self.logger.asset_error(f"Only << file >> or << memory >> is supported for << interface_mode >>")  
        
    def load_summary(self):
        """ Load the already created summary YAML for the user to review.
            
        Args: 
            yaml_dict (dict): loaded summary yaml 

        Returns: -

        """
        yaml_dict = dict()
        yaml_file_path = None 
        if self.asset_envs['pipeline']  == "train_pipeline":
            yaml_file_path = self.asset_envs["artifacts"]["train_artifacts"] + "score/" + "train_summary.yaml" 
        elif self.asset_envs['pipeline'] == "inference_pipeline":
            yaml_file_path = self.asset_envs["artifacts"]["inference_artifacts"] + "score/" + "inference_summary.yaml" 
        else: 
            self.logger.asset_error(f"You have written wrong value for << asset_source  >> in the config yaml file. - { self.asset_envs['pipeline']} \n Only << train_pipeline >> and << inference_pipeline >> are permitted")
        ## check summary yaml existence
        if os.path.exists(yaml_file_path):
            try:
                with open(yaml_file_path, encoding='UTF-8') as f:
                    yaml_dict  = yaml.load(f, Loader=yaml.FullLoader)
            except:
                self.logger.asset_error(f"Failed to call << load_summary>>. \n - summary yaml file path : {yaml_file_path}")
        else: 
            self.logger.asset_error(f"Failed to call << load_summary>>. \n You did not call << save_summary >> in previous assets before calling << load_summary >>")
        ## not return alo internal generated keys  
        try: 
            del yaml_dict['file_path']
            del yaml_dict['version']
            del yaml_dict['date']
        except: 
            self.logger.asset_error(f"[@ load_summary] Failed to delete the key of << file_path, version, date >> in the yaml dictionary.")
        return yaml_dict 
    
    def save_summary(self, result, score, note="", probability={}):
        """ Save train_summary.yaml (when summary is also conducted during train) or inference_summary.yaml.
            e.g. self.asset.save_summary(result='OK', score=0.613, note='alo.csv', probability={'OK':0.715, 'NG':0.135, 'NG1':0.15}
            
        Args: 
            result      (str): Inference result summarized info. (length limit: 25) 
            score:      (float): model performance score to be used for model retraining (0 ~ 1.0)
            note        (str): optional & additional info. for inference result (length limit: 100) (optional)
            probability (dict): probability per class prediction if the solution is classification problem.  (optional)
                                e.g. {'OK': 0.6, 'NG':0.4}

        Returns: 
            summaray_data   (dict): data tobe saved in summary yaml

        """
        result_len_limit = 32
        note_len_limit = 128 
        ## check result length limit 12 
        if not isinstance(result, str) or len(result) > result_len_limit:
            self.logger.asset_error(f"The length of string argument << result >>  must be within {result_len_limit} ")
        ## check score range within 0 ~ 1.0 
        if not isinstance(score, (int, float)) or not 0 <= score <= 1.0:
            self.logger.asset_error("The value of float (or int) argument << score >> must be between 0.0 and 1.0 ")
        ## check note length limit 100  
        if not isinstance(result, str) or len(note) > note_len_limit:
            self.logger.asset_error(f"The length of string argument << note >>  must be within {note_len_limit} ")
        ## check probability type (dict)
        if (probability is not None) and (not isinstance(probability, dict)):
            self.logger.asset_error("The type of argument << probability >> must be << dict >>")
        ## check type - probability key: string,value: float or int  
        if len(probability.keys()) > 0: 
            key_chk_str_set = set([isinstance(k, str) for k in probability.keys()])
            value_type_set = set([type(v) for v in probability.values()])
            if key_chk_str_set != {True}: 
                self.logger.asset_error("The key of dict argument << probability >> must have the type of << str >> ")
            if not value_type_set.issubset({float, int}): 
                self.logger.asset_error("The value of dict argument << probability >> must have the type of << int >> or << float >> ")
            ## check probability values sum = 1 
            if round(sum(probability.values())) != 1: 
                self.logger.asset_error("The sum of probability dict values must be << 1.0 >>")
        else:
            pass 
        # FIXME e.g. 0.50001, 0.49999 case?
        # FIXME it is necessary to check whether the sum of the user-entered dict is 1, anticipating a floating-point error
        def make_addup_1(prob):  
            ## Process the probabilities to sum up to 1, displaying up to two decimal places
            max_value_key = max(prob, key=prob.get) 
            proc_prob_dict = dict()  
            for k, v in prob.items(): 
                if k == max_value_key: 
                    proc_prob_dict[k] = 0 
                    continue
                proc_prob_dict[k] = round(v, 2) 
            proc_prob_dict[max_value_key] = round(1 - sum(proc_prob_dict.values()), 2)
            return proc_prob_dict
        
        if (probability != None) and (probability != {}): 
            probability = make_addup_1(probability)
        else: 
            probability = {}
        ## generate {file_path} 
        file_path = ""     
        ## external save artifacts path 
        if self.save_artifacts_path is None: 
            ## train or inference
            mode = self.asset_envs['pipeline'].split('_')[0] 
            self.logger.asset_warning(f"Please enter the << external_path - save_{mode}_artifacts_path >> in the experimental_plan.yaml.")
        else: 
            file_path = self.save_artifacts_path 
        ## version: string type
        ver = ""
        if self.solution_metadata_version == None: 
            self.solution_metadata_version = ""
        else: 
            ver = 'v' + str(self.solution_metadata_version)  
        ## dict type data to be saved in summary yaml 
        summary_data = {
            'result': result,
            'score': round(score, 2), 
            'date':  datetime.now(timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S'), 
            'note': note,
            'probability': probability,
            'file_path': file_path,  
            'version': ver
        }
        if self.asset_envs['pipeline']  == "train_pipeline":
            file_path = self.asset_envs["artifacts"]["train_artifacts"] + "score/" + "train_summary.yaml" 
        elif self.asset_envs['pipeline'] == "inference_pipeline":
            file_path = self.asset_envs["artifacts"]["inference_artifacts"] + "score/" + "inference_summary.yaml" 
        else: 
            self.logger.asset_error(f"You have written wrong value for << asset_source  >> in the config yaml file. - { self.asset_envs['pipeline']} \n Only << train_pipeline >> and << inference_pipeline >> are permitted")
        ## save summary yaml 
        try:      
            with open(file_path, 'w') as file:
                yaml.dump(summary_data, file, default_flow_style=False)
            self.logger.asset_message(f"Successfully saved inference summary yaml. \n >> {file_path}") 
            self.logger.asset_info(f"Save summary : \n {summary_data}\n")
        except: 
            self.logger.asset_error(f"Failed to save summary yaml file \n @ << {file_path} >>")
             
        return summary_data
    
    def get_model_path(self, use_inference_path=False):
        """ Return model path needed for model load or save 
            
        Args: 
            use_inference_path  (bool): By default it is set to False, so both train and inference pipelines return the train model path. 
                                        Exceptionally, if set to True, the inference pipeline returns the subfolder of its own pipeline name.

        Returns: 
            model_path  (str): model path

        """
        ## FIXME The step names should be paired with the same, with the exception of train-inference pairs. \
        ## However, in cases like cascaded train, it's train1-inference1, train2-inference2 as pairs.
        ## {use_inference_path} type check 
        if not isinstance(use_inference_path, bool):
            self.logger.asset_error("The type of argument << use_inference_path >>  must be << boolean >> ")
        ## check pipeline name 
        allowed_pipeline_mode_list = ["train_pipeline",  "inference_pipeline"]
        current_pipe_mode = self.asset_envs['pipeline']
        if current_pipe_mode not in allowed_pipeline_mode_list: 
            self.logger.asset_error(f"You entered the wrong parameter for << user_parameters >> in your config yaml file : << {current_pipe_mode} >>. \n L ""You can select the pipeline_mode among << {allowed_pipeline_mode_list} >>"" ")
        ## create step path 
        current_step_name = self.asset_envs['step'] 
        ## TODO A discussion is needed to amend the code to read train2 as train as well.
        current_step_name = ''.join(filter(lambda x: x.isalpha() or x == '_', current_step_name))
        artifacts_name = "train_artifacts"
        if use_inference_path == True and current_pipe_mode == "inference_pipeline":
            ## use infernce path
            artifacts_name = "inference_artifacts"
        elif use_inference_path == True and current_pipe_mode != "inference_pipeline":
            self.logger.asset_error("If you set 'use_inference_path' to True, it should operate in the inference pipeline.")
        ## model path  
        model_path = self.asset_envs["artifacts"][artifacts_name] + f"models/{current_step_name}/"
        ## If there is no step with the same name in the train artifacts during the inference pipeline, it results in an error.
        if use_inference_path == False: 
            if (current_pipe_mode  == "inference_pipeline") and (current_step_name != "inference"):
                if not os.path.exists(model_path):
                    self.logger.asset_error(f"You must execute train pipeline first. There is no model path : \n {model_path}") 
            elif (current_pipe_mode  == "inference_pipeline") and (current_step_name == "inference"):
                model_path = self.asset_envs["artifacts"][artifacts_name] + f"models/train/"
                if not os.path.exists(model_path): 
                    self.logger.asset_error(f"You must execute train pipeline first. There is no model path : \n {model_path}") 
                elif (os.path.exists(model_path)) and (len(os.listdir(model_path)) == 0): 
                    self.logger.asset_error(f"You must generate train model first. There is no model in the train model path : \n {model_path}")    
        ## create model path 
        os.makedirs(model_path, exist_ok=True) 
        self.logger.asset_message(f"Successfully got model path for saving or loading your AI model: \n {model_path}")
        return model_path

    def get_output_path(self):
        """ Return the path to save train or inference artifacts output.
            
        Args: -

        Returns: 
            output_path (str): output path 

        """
        allowed_pipeline_mode_list = ["train_pipeline",  "inference_pipeline"]
        current_pipe_mode = self.asset_envs['pipeline']
        if current_pipe_mode  not in allowed_pipeline_mode_list: 
            self.logger.asset_error(f"You entered the wrong parameter for << user_parameters >> in your config yaml file : << {current_pipe_mode} >>. \n - ""You can select the pipeline_mode among << {allowed_pipeline_mode_list} >>"" ")
        ## create output path 
        output_path = ""
        if  current_pipe_mode == "train_pipeline":
            output_path = self.asset_envs["artifacts"]["train_artifacts"] + f"output/"
            os.makedirs(output_path, exist_ok=True) 
        elif current_pipe_mode == 'inference_pipeline': 
            output_path = self.asset_envs["artifacts"]["inference_artifacts"] + f"output/"
            os.makedirs(output_path, exist_ok=True)
        self.logger.asset_message(f"Successfully got << output path >> for saving your data into csv or jpg file: \n {output_path}")
        
        return output_path

    def get_extra_output_path(self):
        """ Return the path to save extra train or inference artifacts output.
            
        Args: -

        Returns: 
            output_path (str): extra output path 

        """
        allowed_pipeline_mode_list = ["train_pipeline",  "inference_pipeline"]
        current_pipe_mode = self.asset_envs['pipeline']
        if current_pipe_mode  not in allowed_pipeline_mode_list: 
            self.logger.asset_error(f"You entered the wrong parameter for << user_parameters >> in your config yaml file : << {current_pipe_mode} >>. \n - ""You can select the pipeline_mode among << {allowed_pipeline_mode_list} >>"" ")
        ## create extra output path 
        extra_output_path = ""
        current_step_name = self.asset_envs['step'] 
        if  current_pipe_mode == "train_pipeline":
            extra_output_path = self.asset_envs["artifacts"]["train_artifacts"] + f"extra_output/{current_step_name}/"
            os.makedirs(extra_output_path, exist_ok=True) # exist_ok =True : 이미 존재하면 그대로 둠 
        elif current_pipe_mode == 'inference_pipeline': 
            extra_output_path = self.asset_envs["artifacts"]["inference_artifacts"] + f"extra_output/{current_step_name}/"
            os.makedirs(extra_output_path, exist_ok=True)
        self.logger.asset_message(f"Successfully got << extra output path >> for saving your output data: \n {extra_output_path} ")
        return extra_output_path
    
    def get_report_path(self):
        """ Return the path to save report (Only supports train pipeline)
            
        Args: -

        Returns: 
            report_path (str): path to save report.html

        """
        allowed_pipeline_mode_list = ["train_pipeline"]
        current_pipe_mode = self.asset_envs['pipeline']
        if current_pipe_mode  not in allowed_pipeline_mode_list: 
            self.logger.asset_error(f"<< get_report_path >> only can be used in << train pipeline >> \n Now: << {current_pipe_mode} >> ")
        ## create report path 
        report_path = self.asset_envs["artifacts"]["train_artifacts"] + "report/"
        os.makedirs(report_path, exist_ok=True)
        self.logger.asset_message(f"Successfully got << report path >> for saving your << report.html >> file: \n {report_path}")
        
        return report_path
    
    #####################################
    ####      INTERNAL FUNCTION      #### 
    #####################################
         
    def _check_config_key(self, prev_config):
        """ Check if the user has not removed any keys from the config loaded from a previous asset within a specific asset.
            
        Args: 
            prev_config (dict): Config saved from the previous asset.

        Returns: -

        """
        ## don't delete already existing keys
        for k in prev_config.keys(): 
            if k not in self.asset_config.keys(): 
                self.logger.asset_error(f"The key << {k} >>  of config dict is deleted in this step. Do not delete key.")  
        
    def _check_data_key(self, prev_data):
        """ Check if the user has not removed any keys from the data loaded from a previous asset within a specific asset.
            
        Args: 
            prev_data (dict): Data saved from the previous asset.

        Returns: -

        """
        ## don't delete already existing keys
        for k in prev_data.keys(): 
            if k not in self.asset_data.keys(): 
                self.logger.asset_error(f"The key << {k} >>  of data dict is deleted in this step. Do not delete key.")  


    def check_args(self, arg_key, is_required=False, default="", chng_type="str" ):
        """ Check user parameter. Replace value & type 
            
        Args: 
            arg_key     (str): user parameter name 
            is_required (bool): the necessity of existence.
            default     (str): the value that will be forcibly entered if user parameters do not exist.
            chng_type   (str): type conversion - list, str, int, float, bool

        Returns: 
            arg_value   (*): converted arg
            
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
        chk_type = type(arg_value)
        if chk_type == list:
            pass
        else:
            arg_value = _convert_variable_type(arg_value, chng_type)
        return arg_value

                
    def _asset_start_info(self):
        """ format asset start information log
            
        Args: -

        Returns: -
            
        """
        ## color: cyan
        msg = "".join(["\033[96m", 
            f"\n=========================================================== ASSET START ===========================================================\n",
            f"- time (UTC)        : {datetime.now(timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"- current step      : {self.asset_envs['step']}\n",
            f"- asset branch.     : {self.asset_branch}\n", 
            f"- alolib ver.       : {self.alolib_version}\n",
            f"- alo ver.          : {self.alo_version}\n",
            f"- load config. keys : {self.asset_config.keys()}\n", 
            f"- load data keys    : {self.asset_data.keys()}\n",
            f"- load args.        : {pformat(self.asset_args, width=200, indent=4)}\n",
            f"====================================================================================================================================\n",
            "\033[0m"])
        self.logger.asset_message(msg)


    def _asset_finish_info(self): 
        """ format asset finish information log
            
        Args: -

        Returns: -
            
        """
        ## color: cyan
        msg = "".join(["\033[96m", 
            f"\n=========================================================== ASSET FINISH ===========================================================\n",
            f"- time (UTC)        : {datetime.now(timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"- current step      : {self.asset_envs['step']}\n",
            f"- save config. keys : {self.asset_config.keys()}\n",
            f"- save data keys    : {self.asset_data.keys()}\n",
            f"====================================================================================================================================\n",
            "\033[0m"])
        self.logger.asset_message(msg)
        
    #####################################
    ####        RUN DECORATOR        #### 
    #####################################
    
    def profile_cpu(self, func):
        """ cpu profiling decorator
            
        Args: 
            func    (function): original function 

        Returns: 
            wrapper (function): wrapped function
            
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            ## CPU usage measure start
            step = self.asset_envs["step"]
            pid = os.getpid()
            ppid = psutil.Process(pid)
            cpu_usage_start = ppid.cpu_percent(interval=None)
            ## call original function 
            result = func(self, *args, **kwargs)  
            ## CPU usage measure finish
            cpu_usage_end = ppid.cpu_percent(interval=None)
            msg = f"- STEP: {step} ~ CPU USAGE (%)        : {cpu_usage_end - cpu_usage_start} %"
            msg = display_resource(step, msg=msg)
            self.logger.asset_message(msg)
            return result
        return wrapper
    

    def decorator_run(func):
        """ A function for a decorator to control the logic of the run function in a user asset.
            e.g. @decorator_run 
            
        Args: 
            func    (function): user asset run function 

        Returns: 
            _run    (function): decorated run functino 
            
        """
        _func = func 
        def _run(self, *args, **kwargs):  
            ## resource check partial decorating 
            if self.asset_envs["check_resource"] == False:
                func = _func
            elif self.asset_envs["check_resource"] == True:                
                ## partial decorator 1 - memory profiler (if on, pipeline run time increases)
                func = profile(precision=2)(_func)
                ## partial decorator 2 - cpu profiler              
                func = self.profile_cpu(func)  
            ## get current step     
            step = self.asset_envs["step"]
            ## (Note) The key [show] is a special key for parsing when creating the tact-time table after ALO run
            self.logger.asset_info(f"{step} asset start", show=True) 
            ## get previvious step data, config      
            prev_data, prev_config = self.asset_data, self.asset_config 
            try:
                ## print asset start info. 
                self._asset_start_info() 
                ## call user asset run  
                func(self, *args, **kwargs)
                ## limit save_data() and save_config() calls to once per step (otherwise, they will be overwritten)
                if (self.asset_envs['save_data'] != 1) or (self.asset_envs['save_config'] != 1):
                        self.logger.asset_error(f"[@ {step} asset] You did not call (or call too many times) the << self.asset.save_data() >> or << self.asset.save_conifg() >> API")
                if (not isinstance(self.asset_data, dict)) or (not isinstance(self.asset_config, dict)):
                    self.logger.asset_error(f"[@ {step} asset] You must input dictionary type argument for << self.asset.save_data()>> or << self.asset.save_config() >>")  
                ## check whether the existing config keys have not been deleted
                self._check_config_key(prev_config)
                ## check whether the existing data keys have not been deleted
                self._check_data_key(prev_data)
                self.logger.asset_info(f"{step} asset finish", show=True) 
            except:
                raise 
            ## print asset finish info.
            self._asset_finish_info()
            return self.asset_data, self.asset_config  
        return _run