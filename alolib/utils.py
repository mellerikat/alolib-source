# -*- coding: utf-8 -*-
import json 
import os
import pickle
import shutil

def load_file(_data_file):
    """ load file and return data
        extension supported: (pkl, json, params, log)

    Args:
        data_file (str) : data file path
                         
    Returns: 
        _data   (dict): loaded data

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
 
def save_file(_data, _data_file):
    """ save data into file.
        extension supported: (pkl, json, params, log)

    Args:
        data    (dict): data tobe saved
        data_file (str): data file saved path
                         
    Returns: -

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
                    ## ensure_ascii=False: suport Korean
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
    """ Check file existence in the directory 

    Args:
        _filename (str) : the file name with full directory 
        remake (bool) : Create the path if not path
                         
    Returns: 
        Bool

    """
    ## if the directory not exist
    if not os.path.exists(os.path.dirname(_filename)):
        os.makedirs(os.path.dirname(_filename))
        return False
    else:
        ## recreate directory 
        if remake == True:
            shutil.rmtree(os.path.dirname(_filename))
            os.makedirs(os.path.dirname(_filename))
        return True
 
def _convert_variable_type(variable, target_type):
    """ convert variable's type into target type 

    Args:
        variable    (*): variable tobe type-converted
        target_type (*): target type
                         
    Returns: 
        type converted variable

    """
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
    """ During the load_data process, return a data dictionary to the user 
        which contains only the keys that include {_partial_load}.

    Args:
        _asset_data (dict)    :  data dict
        _partial_load (str)   : sub-path under the {HOME}/input/{pipline}/ 
                         
    Returns: 
        partial k,v extract from {_asset_data}

    """
    ## partial keys to list 
    partial_key_list = [] 
    for k in _asset_data.keys(): 
        if _partial_load in k:  
            partial_key_list.append(k)
    # partial k,v extract from {_asset_data}
    return dict(filter(lambda item: item[0] in partial_key_list, _asset_data.items()))

def display_resource(step, msg):
    """ format conversion for displaying resource (memory, cpu) 

    Args:
        msg: message tobe printed
                         
    Returns: 
        partial k,v extract from {_asset_data}

    """
    assert type(msg) == str
    ## (Note) mem, cpu decorator in asset.py - decorator_run 
    ## color: bright yellow 
    msg = "".join(["\033[93m",                                     
    f"\n----------------------------------------------------------- Finished displaying < MEMORY > usage ( asset: {step} )\n",
    f"\n {msg} \n",
    f"----------------------------------------------------------- Finished displaying < CPU > usage ( asset: {step} ) \n",
    "\033[0m"])
    return msg