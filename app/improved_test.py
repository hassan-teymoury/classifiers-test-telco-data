import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import unittest
import os
import numpy as np




class TestModel(unittest.TestCase):

        
    
    def setUp(self):
        
        # Load the data from a CSV file
        
        self.data_path = "data.csv"
        self.target_col = "target"
        try:
            self.data = pd.read_csv(self.data_path)
        except:
            self.data = pd.DataFrame()
        
        # Split the data into features and labels
        try:
            self.X = self.data.drop(self.target_col, axis=1)
            self.y = self.data[self.target_col]
        except:
            self.X = pd.DataFrame()
            self.y = pd.DataFrame()
        
        # Split the data into training and testing sets
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        except:
            self.X_train, self.X_test, self.y_train, self.y_test = (pd.DataFrame() for _ in range(4))
        
        # Preprocess the data by scaling the features
        self.scaler = StandardScaler()
        try:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        except:
            self.X_train = np.array(self.X_train)
            self.X_test = np.array(self.X_test)
        
        # Train the logistic regression model
        self.model = LogisticRegression()
        try:
            self.model.fit(self.X_train, self.y_train)
        except:
            pass
        
        # Test the model on the test set
        try:
            self.y_pred = self.model.predict(self.X_test)
        except:
            self.y_pred = None
        
    
    def test_isdata_csv(self):
        condition = self.data_path.split(".")[-1] in ["csv", "txt"] 
        self.assertTrue(condition, 
                        "Your dataframe should be in 'xxxxx.csv' or 'xxxxx.txt' format")
    
    def test_loading_data(self):
        self.assertTrue(os.path.isfile(self.data_path),
                        f"There is no such a directory {self.data_path}")
    
    def test_dataset_labels(self):
        self.assertTrue(self.target_col in self.data.columns.tolist(),
                        f"There is no column with the name '{self.target_col}' in your dataset")
        
    def test_dataset_features(self):
        condition = len(self.data.columns.tolist()) > 1
        self.assertTrue(condition, 
                        "your dataframe should have more than on feature/columns for supervised learning")
    
    
    def test_tarin_test_split(self):
        try:
            ratio_test_train = len(self.X_test) / len(self.X_train)
        except:
            ratio_test_train = np.nan
            
        self.assertAlmostEqual(0.2, ratio_test_train, delta=0.02,
                               msg="Your data is not splitted to test and train with your desired test_size")
    
    
    def test_scaler(self):
        condition = (self.X_train.shape[0] > 0) and (self.X_test.shape[0] > 0)
        self.assertTrue(condition, 
                        msg="Data can not be scaled")
    
    def test_model_has_coefs(self):
        try:
            coef = self.model.coef_
        except:
            coef = None
        self.assertIsNotNone(coef, 
                             msg="Model has not been fitted to the train data and so can not predict")
        
    def test_preds_gen(self):
        self.assertIsNotNone(self.y_pred,
                             msg="There is no prediction array to calculate accuracy")
    
    
    
    def test_accuracy(self):
        # Assert
        expected_accuracy = 0.9
        if self.y_pred is not None:
            actual_accuracy = accuracy_score(self.y_test, self.y_pred)
        else:
            actual_accuracy = -1
            
        self.assertAlmostEqual(actual_accuracy, expected_accuracy, delta=0.05,
                               msg="Model can not get your desired accuracy on test data")

    

if __name__ == '__main__':
    unittest.main()