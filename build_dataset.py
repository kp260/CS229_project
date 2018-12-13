"""Read, split and save the dataset for the model"""

import os
import numpy as np
import pandas as pd


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def data_processing(path):

    # Download data and randomly reshuffle rows. 
    data_set = pd.read_csv(path,header=0)
    data_set = data_set.dropna() # should be complete entries only
    #cols = data_set.columns.drop(['LOAN_ID','Class_orig', 'Class'])
    cols = data_set.columns.drop(['LOAN_ID', 'Class'])
    data_set[cols] = data_set[cols].apply(pd.to_numeric, errors='coerce')
    data_set['Class'] = data_set['Class'].apply(pd.to_numeric, downcast='integer', errors='coerce')
    #data_set['Class_orig'] = data_set['Class_orig'].apply(pd.to_numeric, downcast='integer', errors='coerce')
    data_set = data_set.sample(frac=1).reset_index(drop=True) #reshuffle rows

    
    # Normalize columns  
    cols_to_norm = ['ORIG_RT','ORIG_AMT', 'NUM_BO', 'OLTV','DTI','CSCORE_B', 'ZIP_3', 'MI_PCT']
    data_set[cols_to_norm] = data_set[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))
    


    # Split data into train, dev and test sets
    shape = data_set.shape
    m = shape[0] # number of total examples
    dev_m = np.floor(m*0.05)
    test_m = np.floor(m*0.05)
    train_m = m - test_m - dev_m

    num_class = 6
    
    # Assign X and Y to train, dev and test sets
    X_train = data_set.loc[0:train_m-1,'ORIG_RT':'X12']
    Y_train_orig = data_set.loc[0:train_m-1, 'Class']
    Y_train_orig = pd.DataFrame.as_matrix(Y_train_orig)
    Y_train_orig = Y_train_orig.astype(int)
    
    X_dev = data_set.loc[train_m:train_m+dev_m-1,'ORIG_RT':'X12']
    Y_dev_orig = data_set.loc[train_m:train_m+dev_m-1, 'Class']
    Y_dev_orig = pd.DataFrame.as_matrix(Y_dev_orig)
    Y_dev_orig = Y_dev_orig.astype(int)
    
    X_test = data_set.loc[train_m+dev_m:,'ORIG_RT':'X12']
    Y_test_orig = data_set.loc[train_m+dev_m:, 'Class']
    Y_test_orig = pd.DataFrame.as_matrix(Y_test_orig)
    Y_test_orig = Y_test_orig.astype(int)

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, num_class)
    Y_dev = convert_to_one_hot(Y_dev_orig, num_class)
    Y_test = convert_to_one_hot(Y_test_orig, num_class)
    
    # Reshape Y to (num examples, num classes)
    Y_train = (Y_train.reshape(Y_train.shape[0], -1)).T
    Y_dev = (Y_dev.reshape(Y_dev.shape[0], -1)).T
    Y_test = (Y_test.reshape(Y_test.shape[0], -1)).T
    
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test
    


if __name__ == "__main__":
    
    # Check that the dataset exists 
    path_dataset = 'data/data.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading dataset into memory...")
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = data_processing(path_dataset)
    print("- done.")

    # Split the dataset into train, dev and split 
    train_X_dataset = pd.DataFrame(X_train)
    dev_X_dataset = pd.DataFrame(X_dev)
    test_X_dataset = pd.DataFrame(X_test)
    train_Y_dataset = pd.DataFrame(Y_train)
    dev_Y_dataset = pd.DataFrame(Y_dev)
    test_Y_dataset = pd.DataFrame(Y_test)

    # Save the datasets to files
    train_X_dataset.to_csv(os.path.join("data/train", 'X.csv'))
    dev_X_dataset.to_csv(os.path.join("data/dev", 'X.csv'))
    test_X_dataset.to_csv(os.path.join("data/test", 'X.csv'))
    train_Y_dataset.to_csv(os.path.join("data/train", 'Y.csv'))
    dev_Y_dataset.to_csv(os.path.join("data/dev", 'Y.csv'))
    test_Y_dataset.to_csv(os.path.join("data/test", 'Y.csv'))
  