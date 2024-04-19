# -*- coding: utf-8 -*-
import os
import shutil
import pickle
import json 

def load_file(_data_file):
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
def save_file(_data, _data_file):
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
 

def _convert_variable_type(variable, target_type):
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
            
def _extract_partial_data(_asset_data, _partial_load):             
    """ Description
        -----------
            - load_data 시 사용자에게 반환되는 data dict의 key 중 _partial_load를 담고 있는 것만 추려서 data dict를 반환합니다. 
            
        Parameters
        -----------
            - _asset_data (dict)    :  data dict
            - _partial_load (str)   : {HOME}/input/{pipline}/ 하위에 존재하는 경로 
            
        Return
        -----------
            - _asset_data           : 부분적으로 추려진 asset data 
            
        Example
        -----------
            - asset_data = _extract_partial_data(_asset_data, _partial_load)
    """
    # 부분적으로 추릴 key를 list화 
    partial_key_list = [] 
    for k in _asset_data.keys(): 
        if _partial_load in k:  
            partial_key_list.append(k)
    # partial k,v extract from asset_data
    return dict(filter(lambda item: item[0] in partial_key_list, _asset_data.items()))

def display_resource(step, msg):
    '''
    - msg: printed message (cpu usage)
    ''' 
    assert type(msg) == str
    # asset.py - decorator_run 에 mem, cpu decorator 붙어있는 점 활용
    msg = "".join(["\033[93m", #bright yellow                                     
    f"\n----------------------------------------------------------- Finished displaying < MEMORY > usage ( asset: {step} )\n",
    f"\n {msg} \n",
    f"----------------------------------------------------------- Finished displaying < CPU > usage ( asset: {step} ) \n",
    "\033[0m"])
    return msg