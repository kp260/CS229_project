"""Define the model."""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn



# ------------------------------------------------
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (number of examples, n_x)
    parameters -- python dictionary containing your parameters "W1", "b1"
                  the shapes are given in initialize_parameters

    Returns:
    Z1 -- the output of the LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
            
    Z1 = tf.add(tf.matmul(X, W1),b1)                                          
                                                                                                  
    return Z1
#----------------------------------------------
def forward_propagation_deep(X, parameters, is_training, keep_prob):
    """
    Implements the forward propagation for the model: [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (number of examples, n_x)
    parameters -- python dictionary containing parameters 
                  the shapes are given in initialize_parameters

    Returns:
    Z -- the output of the LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    
    if is_training:
        keep_prob = keep_prob
    else: keep_prob = 1
    
    X = tf.nn.dropout(X, keep_prob)
    Z1 = tf.add(tf.matmul(X, W1),b1) 
    # batch normalization applied for training only
    #Z1 = tf.layers.batch_normalization(Z1, training = is_training)                                        
    
    A1 = tf.nn.relu(Z1) 
    A1 = tf.nn.dropout(A1, keep_prob)                                            
    Z2 = tf.add(tf.matmul(A1,W2),b2)
    
    #Z2 = tf.layers.batch_normalization(Z2, training = is_training)                                               
    A2 = tf.nn.relu(Z2)    
    A2 = tf.nn.dropout(A2, keep_prob)                                          
    Z3 = tf.add(tf.matmul(A2, W3), b3) 
    
    #Z3 = tf.layers.batch_normalization(Z3, training = is_training)
    A3 = tf.nn.relu(Z3)    
    A3 = tf.nn.dropout(A3, keep_prob)                                          
    Z4 = tf.add(tf.matmul(A3, W4), b4) 
    
    #Z4 = tf.layers.batch_normalization(Z4, training = is_training)
    A4 = tf.nn.relu(Z4) 
    Z5 = tf.add(tf.matmul(A4, W5), b5)    
    A5 = tf.nn.relu(Z5) 
    Z6 = tf.add(tf.matmul(A5, W6), b6)    
                      
                                                                                                                       
    return Z6
#----------------------------------------------
def forward_propagation_cnn(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (number of examples, input_size)
    parameters -- python dictionary containing parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 1x3, sride 2, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,1,3,1], strides = [1,1,2,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 1x2, stride 2, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,1,2,1], strides = [1,1,2,1], padding = 'SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function.
    # 7 neurons in output layer.  
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
    

    return Z3
#---------------------------------------
def initialize_parameters(n_x, n_y):
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [n_x, n_y]
                        b1 : [1, n_y]
               
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1
    """
 
    W1 = tf.get_variable("W1", [n_x, n_y], initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    b1 = tf.get_variable("b1", [1, n_y], initializer = tf.zeros_initializer(), dtype=tf.float32)


    parameters = {"W1": W1,
                  "b1": b1}
    
    return parameters
#--------------------------------------------------
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1,layer_dims[l])
    """
     
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layer_dims[l-1], layer_dims[l]], initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [1, layer_dims[l]], initializer = tf.zeros_initializer(), dtype=tf.float32)

        assert(parameters['W' + str(l)].shape == (layer_dims[l-1], layer_dims[l]))
        assert(parameters['b' + str(l)].shape == (1,layer_dims[l]))

        
    return parameters
#-------------------------------------------------
def initialize_parameters_cnn():
    """
    Initializes weight parameters to build a neural network with tensorflow. n_H=1
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
   
    
    init_xavier = tf.contrib.layers.xavier_initializer()
    W1 = tf.get_variable("W1", [1,5,1,32], initializer = init_xavier)
    W2 = tf.get_variable("W2", [1,3,32,64], initializer = init_xavier)
    

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters
# ------------------------------------------------

    """
    Initializes weight parameters to build a neural network with tensorflow. 
    Returns:
    parameters -- a dictionary of tensors containing W, b
    """
     
    init_xavier = tf.random_normal_initializer()              
    W_output = tf.get_variable("Wo", [num_units,n_classes], initializer = init_xavier)
    b_output = tf.get_variable("bo", [n_classes], initializer = init_xavier)
    
    W_hidden = tf.get_variable("Wh",[n_inputs,n_classes], initializer = init_xavier)
    b_hidden = tf.get_variable("bh", [n_classes], initializer = init_xavier)

    parameters = {"Wh": W_hidden,
                  "bh": b_hidden,
                  "Wo": W_output,
                  "bo": b_output,}
    
    return parameters
    
def build_model(mode, X, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    is_training = (mode == 'train')
    if params.model_version == 'baseline':
        # baseline model implementation
        
        n_y = params.class_num
        n_x = params.features_len
        parameters = initialize_parameters(n_x, n_y)
        logits = forward_propagation(X, parameters)
        
    elif params.model_version == 'deeplayer':
        # deep neural network
        keep_prob = 1 - params.dropout_rate
        n_y = params.class_num
        n_x = params.features_len
        layer_dims = params.layers_dim
        parameters = initialize_parameters_deep(layer_dims)
        logits = forward_propagation_deep(X, parameters, is_training, keep_prob)

    elif params.model_version == 'convnet':
        # convolutional neural network
        X = tf.expand_dims(tf.expand_dims(X, 1),3) # change shape from (num examples, num features) to (nun_examples, 1, num_features, 1)
        parameters = initialize_parameters_cnn()
        logits = forward_propagation_cnn(X, parameters)
    
      
    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return logits
#-------------------------------------------
def alpha_c(labels, params):
    # median/freq_c
    
    coefficients = tf.placeholder(dtype = tf.float32, shape=[params.batch_size, 1])
    alpha_coef = tf.placeholder(dtype = tf.float32, shape=[6,1])
    
    class_freq = np.zeros((6,1))
    class_freq[0,:] = params.m0
    class_freq[1,:] = params.m1
    class_freq[2,:] = params.m2
    class_freq[3,:] = params.m3
    class_freq[4,:] = params.m4
    class_freq[5,:] = params.m5
 
    
    class_vec = np.divide(class_freq,params.batch_size)
    class_vec = class_vec + 1e-10
    median_class = class_vec[0,:]
    alpha_coef = np.divide(median_class,class_vec)
    coefficients = tf.matmul(labels, tf.cast(alpha_coef, tf.float32))
    coefficients = tf.convert_to_tensor(coefficients)
    coefficients = tf.expand_dims(coefficients, 1)
    
    
    return coefficients
#-------------------------------------------------
    
def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    X = inputs['features']
    X = tf.cast(X, tf.float32)
    labels = tf.cast(labels, tf.float32)
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, X, params)
        predictions = tf.argmax(tf.nn.softmax(logits), axis=1)
        


    # Define loss and accuracy: apply weights to penalize more for classes with smaller number of samples as to avoid predicting 0 all the time 
    if is_training:
        coefficients = alpha_c(labels, params)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels)
        weighted_losses = unweighted_losses * coefficients
        loss = tf.reduce_mean(weighted_losses)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1) , predictions), tf.float32))
    else: 
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels)
        loss = tf.reduce_mean(unweighted_losses)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1) , predictions), tf.float32))
    
    # Define F1 scores for each class
    confusion_matrix = tf.confusion_matrix(labels=tf.argmax(labels, axis=1), predictions=predictions)
    f1_0 = 2*confusion_matrix[0, 0]/(tf.reduce_sum(confusion_matrix[0,:])+tf.reduce_sum(confusion_matrix[:,0]))
    f1_1 = 2*confusion_matrix[1, 1]/(tf.reduce_sum(confusion_matrix[1,:])+tf.reduce_sum(confusion_matrix[:,1]))
    f1_2 = 2*confusion_matrix[2, 2]/(tf.reduce_sum(confusion_matrix[2,:])+tf.reduce_sum(confusion_matrix[:,2]))
    f1_3 = 2*confusion_matrix[3, 3]/(tf.reduce_sum(confusion_matrix[3,:])+tf.reduce_sum(confusion_matrix[:,3]))
    f1_4 = 2*confusion_matrix[4, 4]/(tf.reduce_sum(confusion_matrix[4,:])+tf.reduce_sum(confusion_matrix[:,4]))
    f1_5 = 2*confusion_matrix[5, 5]/(tf.reduce_sum(confusion_matrix[5,:])+tf.reduce_sum(confusion_matrix[:,5]))
    
    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)
        

  
    
    
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
   
    with tf.variable_scope("metrics"):
        
              
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions),
            'loss': tf.metrics.mean(loss),
            'recall_0': tf.metrics.recall(labels=tf.equal(tf.argmax(labels, axis=1), 0), predictions=tf.equal(predictions,0)),
            'precision_0': tf.metrics.precision(labels=tf.equal(tf.argmax(labels, axis=1), 0), predictions=tf.equal(predictions,0)),
            'recall_1': tf.metrics.recall(labels=tf.equal(tf.argmax(labels, axis=1), 1), predictions=tf.equal(predictions,1)),
            'precision_1': tf.metrics.precision(labels=tf.equal(tf.argmax(labels, axis=1), 1), predictions=tf.equal(predictions,1)),
            'recall_2': tf.metrics.recall(labels=tf.equal(tf.argmax(labels, axis=1), 2), predictions=tf.equal(predictions,2)),
            'precision_2': tf.metrics.precision(labels=tf.equal(tf.argmax(labels, axis=1), 2), predictions=tf.equal(predictions,2)),
            'recall_3': tf.metrics.recall(labels=tf.equal(tf.argmax(labels, axis=1), 3), predictions=tf.equal(predictions,3)),
            'precision_3': tf.metrics.precision(labels=tf.equal(tf.argmax(labels, axis=1), 3), predictions=tf.equal(predictions,3)),
            'recall_4': tf.metrics.recall(labels=tf.equal(tf.argmax(labels, axis=1), 4), predictions=tf.equal(predictions,4)),
            'precision_4': tf.metrics.precision(labels=tf.equal(tf.argmax(labels, axis=1), 4), predictions=tf.equal(predictions,4)),
            'recall_5': tf.metrics.recall(labels=tf.equal(tf.argmax(labels, axis=1), 5), predictions=tf.equal(predictions,5)),
            'precision_5': tf.metrics.precision(labels=tf.equal(tf.argmax(labels, axis=1), 5), predictions=tf.equal(predictions,5)),
            
            'class_0_TP': tf.metrics.true_positives(labels=tf.equal(tf.argmax(labels, axis=1), 0), predictions=tf.equal(predictions,0)),
            'class_0_FP': tf.metrics.false_positives(labels=tf.equal(tf.argmax(labels, axis=1), 0), predictions=tf.equal(predictions,0)),
            'class_0_TN': tf.metrics.true_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 0), predictions=tf.equal(predictions,0)),
            'class_0_FN': tf.metrics.false_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 0), predictions=tf.equal(predictions,0)),
            
            'class_1_TP': tf.metrics.true_positives(labels=tf.equal(tf.argmax(labels, axis=1), 1), predictions=tf.equal(predictions,1)),
            'class_1_FP': tf.metrics.false_positives(labels=tf.equal(tf.argmax(labels, axis=1), 1), predictions=tf.equal(predictions,1)),
            'class_1_TN': tf.metrics.true_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 1), predictions=tf.equal(predictions,1)),
            'class_1_FN': tf.metrics.false_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 1), predictions=tf.equal(predictions,1)),
            
            'class_2_TP': tf.metrics.true_positives(labels=tf.equal(tf.argmax(labels, axis=1), 2), predictions=tf.equal(predictions,2)),
            'class_2_FP': tf.metrics.false_positives(labels=tf.equal(tf.argmax(labels, axis=1), 2), predictions=tf.equal(predictions,2)),
            'class_2_TN': tf.metrics.true_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 2), predictions=tf.equal(predictions,2)),
            'class_2_FN': tf.metrics.false_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 2), predictions=tf.equal(predictions,2)),
            
            'class_3_TP': tf.metrics.true_positives(labels=tf.equal(tf.argmax(labels, axis=1), 3), predictions=tf.equal(predictions,3)),
            'class_3_FP': tf.metrics.false_positives(labels=tf.equal(tf.argmax(labels, axis=1), 3), predictions=tf.equal(predictions,3)),
            'class_3_TN': tf.metrics.true_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 3), predictions=tf.equal(predictions,3)),
            'class_3_FN': tf.metrics.false_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 3), predictions=tf.equal(predictions,3)),
            
            'class_4_TP': tf.metrics.true_positives(labels=tf.equal(tf.argmax(labels, axis=1), 4), predictions=tf.equal(predictions,4)),
            'class_4_FP': tf.metrics.false_positives(labels=tf.equal(tf.argmax(labels, axis=1), 4), predictions=tf.equal(predictions,4)),
            'class_4_TN': tf.metrics.true_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 4), predictions=tf.equal(predictions,4)),
            'class_4_FN': tf.metrics.false_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 4), predictions=tf.equal(predictions,4)),
            
            'class_5_TP': tf.metrics.true_positives(labels=tf.equal(tf.argmax(labels, axis=1), 5), predictions=tf.equal(predictions,5)),
            'class_5_FP': tf.metrics.false_positives(labels=tf.equal(tf.argmax(labels, axis=1), 5), predictions=tf.equal(predictions,5)),
            'class_5_TN': tf.metrics.true_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 5), predictions=tf.equal(predictions,5)),
            'class_5_FN': tf.metrics.false_negatives(labels=tf.equal(tf.argmax(labels, axis=1), 5), predictions=tf.equal(predictions,5))
            
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('f1_0', f1_0)
    tf.summary.scalar('f1_1', f1_1)
    tf.summary.scalar('f1_2', f1_2)
    tf.summary.scalar('f1_3', f1_3)
    tf.summary.scalar('f1_4', f1_4)
    tf.summary.scalar('f1_5', f1_5)
   

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(),tf.local_variables_initializer(),  tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec["true_values"] = tf.argmax(labels, axis=1)
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
 
    

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
