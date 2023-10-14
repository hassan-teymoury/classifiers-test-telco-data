import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import recall_score, accuracy_score
from sklearn.preprocessing import minmax_scale, scale, maxabs_scale, robust_scale, \
    MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from typing import Union, Optional
import scikitplot as skplt
# import torch.nn as nn
import matplotlib.pyplot as plt
import argparse



def parse_configs():
    
    parser = argparse.ArgumentParser(
        description="A custom module for and training/testing some classifier algorithms on Telco customer churn dataset"
        )
    parser.add_argument("--datapath", type=str,
                        help="pass '.csv' dataset path here",
                        default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    parser.add_argument("--topk_features", type=int, 
                        help="choose the number of best features selected by the feature selection process",
                        default=7)
    
    parser.add_argument("--test_size", type=float, 
                        help="Test size used to split data to train/test datasets for validation")
    
    parser.add_argument("--scaler_type", type=str, 
                        help="scaler type used for scaling data",
                        choices=["default_scaler", "minmax_scaler", "robust_scaler", "maxabs_scaler"],
                        default="default_scaler")
    parser.add_argument("--scale_axis", type=int, choices=[0,1],
                        help="scale axis for scaling, 0 for scaling rows and 1 for columns",
                        default=1)
    
    parser.add_argument("--classifier", type=str,
                        help="classifier algorithm you want to use",
                        choices=["xgb", "logistic_regression", "sgd_classifier", "random_forest"],
                        default="xgb")
    
    args = parser.parse_args()
    return args




# Processor object
class DataProcessor:
    def __init__(self, datapath:str, num_best_feats:int, test_size:float, scaler_type:str=None, scale_axis:int=None) -> None:
        """
        parameters:
            datapath: Path to your .csv dataset ---> example: 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
            
            num_best_feats: Number of best features which you want to consider in training process
                            after feature selection
            test_size: Test size is a float number between 0 and 1 and is needed to split the data
                       into train and test data
            scaler_type: Type of the scaling process. options: 
                        'minmax_scaler', 'default_scaler', 'robust_scaler', 'maxabs_scaler', None
            scale_axis: set this parameter when you want scaling on data
                        scaling on rows: axis = 0
                        scaling on columns: axis = 1
                        
        return args:
            numpy.ndarrays: X_train, X_test, y_train, y_test
        """
        
        self.datapath = datapath
        self.dataset = pd.read_csv(datapath)
        self.dataset.drop(["customerID"], axis=1, inplace=True)
        self.dataset.replace(["", " "], np.nan, inplace=True)
        self.meta_data_for_test = {}
        self.target_col = "Churn"
        self.feature_cols = sorted([col for col in list(self.dataset.columns) if col!=self.target_col])
        self.topk = num_best_feats
        self.test_size = test_size
        self.scaler_type = scaler_type
        self.scale_axis = scale_axis
        if self.scale_axis:
            if self.scale_axis > 1:
                self.scale_axis = 1
            elif self.scale_axis < 0:
                self.scale_axis = 0
        else:
            self.scale_axis = 1
        self.forward()
        
    def convert_to_num(self):
        cols = self.dataset.columns
        for col in cols:
            if self.dataset[col].dtype not in (int, float, bool):
                try:
                    self.dataset[col] = self.dataset[col].astype(float)
                    min_val = self.dataset[col].min()
                    self.dataset[col].fillna(min_val-9999, inplace=True)
                except:
                    unique_vals = sorted(self.dataset[col].unique().tolist())
                    unique_map = {val:i for i, val in enumerate(unique_vals)}
                    self.meta_data_for_test[col] = unique_map
                    
                    self.dataset[col].replace(list(unique_map.keys()), 
                                    list(unique_map.values()),
                                    inplace=True)
                    
                    self.dataset[col].fillna(-1, inplace=True)
            else:
                self.dataset[col] = self.dataset[col].astype(float)
                min_val = self.dataset[col].min()
                self.dataset[col].fillna(min_val-9999, inplace=True)
    
    
    def feature_selection(self):
        
        if self.topk > len(self.feature_cols):
            self.topk = len(self.feature_cols)
            
        X = np.array(self.dataset[self.feature_cols])
        # print(self.dataset[self.feature_cols].head())
        y = np.array(self.dataset[self.target_col])
        importances = mutual_info_classif(X, y, random_state=1)
        importance_dict = {feat:importances[i] for i, feat in enumerate(self.feature_cols)}
                
            
        self.sorted_features = {k:importance_dict[k] for k in
                                sorted(importance_dict, key=lambda x:importance_dict[x], reverse=True)}
        self.best_feats = list(self.sorted_features.keys())[:self.topk]
        
    def forward(self):
        self.convert_to_num()
        self.feature_selection()
        X = np.array(self.dataset[self.best_feats])
        y = np.array(self.dataset[self.target_col])
        
        if self.scaler_type:
            if self.scaler_type == "default_scaler":
                self.scaler_obj = StandardScaler()
                
            elif self.scaler_type == "minmax_scaler":
                self.scaler_obj = MinMaxScaler()
                
            elif self.scaler_type == "robust_scaler":
                self.scaler_obj = RobustScaler()
                
            elif self.scaler_type == "maxabs_scaler":
                self.scaler_obj = MaxAbsScaler()
        
        X = self.scaler_obj.fit_transform(X,{"axis":self.scale_axis})
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y , test_size=self.test_size, random_state=1, stratify=y)
       
        
        
class Classifier:
    def __init__(self, dataprocessor:DataProcessor, classifier:str, recall_avg_mode=None) -> None:
        self.classifier = classifier
        self.dataprocessor = dataprocessor
        self.x_test = self.dataprocessor.X_test
        self.y_test = self.dataprocessor.y_test
        self.x_train = self.dataprocessor.X_train
        self.y_train = self.dataprocessor.y_train
        
        if not self.classifier:
            self.classifier = "xgb"
            
        self.accuracy = None
        self.recall = None
        self.recall_avg_mode = recall_avg_mode
        self.test_preds = None
        
        self.test_class_1_num = self.y_test.tolist().count(1)
        self.test_class_0_num = self.y_test.tolist().count(0)
        self.preds_class_1_num = None
        self.preds_class_0_num = None
        
        
    def fit(self):
        if self.classifier == "xgb":
            print(f"{self.classifier} initiating ...")
            
            param = {}
            param['booster'] = 'gbtree'
            param['objective'] = 'binary:logistic'
            param['learning_rate'] = 0.05
            param['gamma'] = 0.5
            param['max_depth'] = 6
            param['n_estimators']=20
            param['max_delta_step'] = 0
            param['base_score'] = 0.5
            
            self.model = xgb.XGBClassifier()
            self.model.set_params(**param)
            self.model.fit(self.dataprocessor.X_train, self.dataprocessor.y_train)
            
        
        elif self.classifier == "logistic_regression":
            print(f"{self.classifier} initiating ...")
            self.model = LogisticRegression(penalty="l2", random_state=1, max_iter=100)
            self.model.fit(self.x_train, self.y_train)

        
        elif self.classifier == "sgd_classifier":
            print(f"{self.classifier} initiating ...")
            self.model = SGDClassifier(loss="log_loss", learning_rate="adaptive",
                                       penalty="l2", eta0=0.05, random_state=1)
            self.model.fit(self.x_train, self.y_train)
        
        elif self.classifier == "random_forest":
            print(f"{self.classifier} initiating ...")
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                                criterion="log_loss",random_state=1)
            self.model.fit(self.x_train, self.y_train)
            
        else:
            print("Sorry, we can not support your desired classifier here")
        
        
        
    
    def evaluate(self):
        preds = self.model.predict(self.x_test)
        self.preds_prob = self.model.predict_proba(self.x_test)
        
        self.test_preds = preds
        self.test_class_0_num = preds.tolist().count(0)
        self.test_class_1_num = preds.tolist().count(1)
        self.accuracy = accuracy_score(y_true=self.y_test,
                                y_pred=preds)
        self.recall = recall_score(y_true=self.y_test,
                                y_pred=preds, average=self.recall_avg_mode)
        
        print("*************** Dataset Info *******************")
        print(f"Number od samples in Training dataset: {self.x_train.shape[0]}\n")
        
        print(f"Number od samples in Training dataset: {self.x_test.shape[0]}\n")
        
        print(f"Categorical mapping dictionary used for entire dataset:\n {self.dataprocessor.meta_data_for_test}\n\n")
        
        print(f"""Number od samples for each class in Training dataset:
              class 0 (not churn): {self.y_train.tolist().count(0)}
              class 1 (churn): {self.y_train.tolist().count(1)}\n
              """)
        
        print(f"""Number od samples for each class in Test dataset:
              class 0 (not churn): {self.y_test.tolist().count(0)}
              class 1 (churn): {self.y_test.tolist().count(1)}\n
              """)
        
        print(f"The accuracy of {self.classifier} for test data: {self.accuracy}\n")
        print(f"The recall score of {self.classifier} for test data: {self.recall}\n")
        
        
        skplt.metrics.plot_confusion_matrix(self.y_test, self.test_preds, normalize=False,
                                            title = f'Confusion Matrix for {self.classifier}')
        
        # fig.add_subplot(sub)
        skplt.metrics.plot_roc(self.y_test, self.preds_prob,
                               title = f'ROC Plot for {self.classifier}')
        
        skplt.metrics.plot_precision_recall(self.y_test, self.preds_prob,
                                            title = f'PR Curve for {self.classifier}')
        plt.show()
        
        
          
        


if __name__ == "__main__":
    args = parse_configs()
    datapath = args.datapath
    num_best_feats = args.topk_features
    test_size = args.test_size
    scaler_type = args.scaler_type
    scale_axis = args.scale_axis
    classifier = args.classifier
    
    processor = DataProcessor(datapath=datapath,
                          num_best_feats=num_best_feats,
                          test_size=test_size,
                          scaler_type=scaler_type,
                          scale_axis=scale_axis)
    
    model = Classifier(
        dataprocessor=processor, classifier=classifier
    )
    
    model.fit()
    model.evaluate()