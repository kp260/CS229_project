"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.evaluation import evaluate
from model.input_fn import input_fn
from model.input_fn import load_Xdataset_from_text, load_Ydataset_from_text
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
    params.update(json_path)


    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Get paths for vocabularies and dataset
    path_eval = os.path.join(args.data_dir, 'test/')

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_X = load_Xdataset_from_text(os.path.join(path_eval, 'X.csv'))
    test_labels = load_Ydataset_from_text(os.path.join(path_eval, 'Y.csv'))

    # Specify other parameters for the dataset and the model
    params.eval_size = params.test_size
    

    # Create iterator over the test set
    inputs = input_fn('eval', test_X, test_labels, params)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', inputs, params, reuse=False)
    logging.info("- done.")

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
