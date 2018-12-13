"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import pandas as pd
import gc

def load_Xdataset_from_text(path_csv):
    """Create tf.data Instance from csv file

    Args:
        path_cvs: (string) path containing one example per line
        

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load csv file  
    data_set = pd.read_csv(path_csv,header=0)
    dataset = data_set.loc[:,'ORIG_RT':'X12' ]
    
   
    


    return dataset

def load_Ydataset_from_text(path_csv):
    """Create tf.data Instance from csv file

    Args:
        path_cvs: (string) path containing one example per line
        

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load csv file
    data_set = pd.read_csv(path_csv,header=0)
    dataset = data_set.loc[:, "0":"5"]


    return dataset

def balanced_generator(X, Y, params, mode):
    """Creates a balanced dataset where each class is equally represented

    Args:
        mode: (string) 'train', 'eval' 
                     At training, shuffle the data and have multiple epochs
        X: (tf.Dataset) yielding list of features
        Y: (tf.Dataset) yielding list of labels/classes
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    
    dataset = tf.concat([X, Y], axis=1)
    
    # Load all the dataset in memory for shuffling if training
    is_training = (mode == 'train')
    
    if is_training:
        
        # Construct datasets containing elements from each class    #27
        idx_0 = tf.where(tf.equal(tf.argmax(dataset[:, params.features_len:(params.features_len + params.class_num)], axis=1), 0))
        idx_0 = tf.random_shuffle(idx_0)#shuffle
        m_0 = params.m1
        i_0 = tf.gather(dataset, idx_0[0:m_0], axis = 0)
        i_0 = tf.squeeze(i_0,1)
        
         
        idx_1 = tf.where(tf.equal(tf.argmax(dataset[:, params.features_len:(params.features_len + params.class_num)], axis=1), 1))
        i_1 = tf.gather(dataset, idx_1, axis = 0)
        i_1 = tf.squeeze(i_1,1)
        i_1 = tf.random_shuffle(i_1) #shuffle
        m_1 = params.m1*15
        n_1 = tf.cast(tf.ceil(m_0/m_1), tf.int32)
        i_1 = tf.tile(i_1, [n_1,1])
        i_1 = tf.slice(i_1, [0,0], [m_0, (params.features_len + params.class_num)])
        
        
        idx_2 = tf.where(tf.equal(tf.argmax(dataset[:, params.features_len:(params.features_len + params.class_num)], axis=1), 2))
        i_2 = tf.gather(dataset, idx_2, axis = 0)
        i_2 = tf.squeeze(i_2,1)
        i_2 = tf.random_shuffle(i_2) #shuffle
        m_2 = params.m2
        n_2 = tf.cast(tf.ceil(m_0/m_2), tf.int32)
        i_2 = tf.tile(i_2, [n_2,1])
        i_2 = tf.slice(i_2, [0,0], [m_0, (params.features_len + params.class_num)])
        
        idx_3 = tf.where(tf.equal(tf.argmax(dataset[:, params.features_len:(params.features_len + params.class_num)], axis=1), 3))
        i_3 = tf.gather(dataset, idx_3, axis = 0)
        i_3 = tf.squeeze(i_3,1)
        i_3 = tf.random_shuffle(i_3) #shuffle
        m_3 = params.m3
        n_3 = tf.cast(tf.ceil(m_0/m_3), tf.int32)
        i_3 = tf.tile(i_3, [n_3,1])
        i_3 = tf.slice(i_3, [0,0], [m_0, (params.features_len + params.class_num)])
        
        idx_4 = tf.where(tf.equal(tf.argmax(dataset[:, params.features_len:(params.features_len + params.class_num)], axis=1), 4))
        i_4 = tf.gather(dataset, idx_4, axis = 0)
        i_4 = tf.squeeze(i_4,1)
        i_4 = tf.random_shuffle(i_4) #shuffle
        m_4 = params.m4
        n_4 = tf.cast(tf.ceil(m_0/m_4), tf.int32)
        i_4 = tf.tile(i_4, [n_4,1])
        i_4 = tf.slice(i_4, [0,0], [m_0, (params.features_len + params.class_num)])
        
        idx_5 = tf.where(tf.equal(tf.argmax(dataset[:, params.features_len:(params.features_len + params.class_num)], axis=1), 5))
        i_5 = tf.gather(dataset, idx_5, axis = 0)
        i_5 = tf.squeeze(i_5,1)
        i_5 = tf.random_shuffle(i_5) #shuffle
        m_5 = params.m5
        n_5 = tf.cast(tf.ceil(m_0/m_5), tf.int32)
        i_5 = tf.tile(i_5, [n_5,1])
        i_5 = tf.slice(i_5, [0,0], [m_0, (params.features_len + params.class_num)])
        
        # reshape so that one element from each of 6 datasets is selected in turn
        ds = tf.reshape(tf.stack([i_0, i_1, i_2, i_3, i_4, i_5], axis=1), [-1, tf.shape(i_0)[1]])
        
        ds = tf.reshape(ds,[params.class_num*m_0, (params.features_len + params.class_num)])
        ds = tf.data.Dataset.from_tensors(ds)

        gc.collect()
    else: ds = tf.data.Dataset.from_tensors(dataset)
    
    return ds



def input_fn(mode, X, Y, params):
    """Input function for consumer_lending model

    Args:
        mode: (string) 'train', 'eval' 
                     At training, shuffle the data and have multiple epochs
        X: (tf.Dataset) yielding list of features
        Y: (tf.Dataset) yielding list of labels/classes
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    dataset = balanced_generator(X,Y, params, mode)
    print(mode, dataset)
    dataset = dataset.apply(tf.contrib.data.unbatch())
    
    # Create batches 
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
    dataset = dataset.prefetch(1)
    
    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    next_example = iterator.get_next()
    #next_example = tf.squeeze(next_example, 0)
    
    init_op = iterator.initializer
    
    X = next_example[:, 0:params.features_len]
    Y = next_example[:, params.features_len:(params.features_len + params.class_num)]

    # Build and return a dictionary containing the nodes / ops
    inputs = {
        'features': X,
        'labels': Y,
        'iterator_init_op': init_op
    }

    return inputs
