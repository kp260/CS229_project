# mortgage_payment_status

Six multi-classification models have been build to predict residential mortgage payment delinquency status.

To run the code for the Baseline, DeepNN and ConvNet follow the following steps.

1. Download the dataset (acquisition and performance txt files) from FINMA website, https://loanperformancedata.fanniemae.com/lppub/index.html, and save it under the data/ directory. Run R script to produce a csv file. 

2. Build the balanced dataset. Run the following script:
python build_dataset.py

This script would add X.csv and Y.csv files to train, dev and test folders.
You are now ready to train the model.

3. Run the following script to start training - model versions are baseline, deeplayer and convnet:
python train.py --model_dir experiments/baseline

4. Evaluate model on the test set, model versions are baseline, deeplayer and convnet:
python evaluate.py --model_dir experiments/basemodel

5. Hyperparameter search:
python search_hyperparams.py --parent_dir experiments/learning_rate

The code for 1-Nearest Neighbour (1NNC.ipynb), RFC (Random Forests.ipynb) and SVM (SVM.ipynb) models is located in the respective jupyter notebooks, which can be run after steps 1-2 from above have been completed.

t-SNE analysis is done in "t-SNE and results tables.ipynb".
