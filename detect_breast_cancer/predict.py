from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Union, List
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from detect_breast_cancer.processing.data_manager import load_dataset

from detect_breast_cancer.config.core import config, BASE_PATH, MODEL_PATH

def make_prediction(input_data: Union[str, List]):
    input_data = input_data if isinstance(input_data, list) else [input_data]
    if isinstance(input_data[0], str):
        X = ImageDataGenerator().flow_from_dataframe(
            pd.DataFrame(input_data), class_mode=None, x_col=0,
            target_size=config.IMAGE_SIZE)
    else:
        X = input_data
    
    with open(BASE_PATH / 'VERSION') as version_file:
        version = version_file.read().strip().replace('.','_')
        model_file = 'model' + version + '.h5'
        labels_file = 'labels' + version + '.pkl'
        
    model = load_model(MODEL_PATH / model_file)
    
    y = model.predict(X)
    with open(MODEL_PATH / labels_file, 'rb') as file:
        labels = pickle.load(file)
        
    return [labels[i].capitalize() for i in (y.ravel()>0.5).astype('int32')]

def print_validation():
    validation = load_dataset(subset='validation')
    lst = [(X.numpy(),y.numpy()) for X,y in validation]
    X = np.concatenate([l[0] for l in lst])
    
    y = np.concatenate([l[1] for l in lst])
    labels = [lab.capitalize() for lab in validation.class_names]
    y = [labels[i] for i in y.ravel().astype(int)]
    
    yp = make_prediction(X)
    
    print(pd.concat([pd.crosstab(y,yp, rownames=['ACTUAL'], colnames=[''])], 
                    keys=['PREDICTION'], axis=1).to_string())

if __name__ =='__main__':
    print_validation()
    