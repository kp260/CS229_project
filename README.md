# mortgage_payment_status
Four multi-classification models have been build to predict residential mortgage payment delinquency status.
To run the code follow the following steps.

1. Download the dataset (acquisition and performance txt files) from FINMA website, https://loanperformancedata.fanniemae.com/lppub/index.html, and save it under the data/ directory. Run R script to produce a csv file. 

2. Build the balanced dataset. Run the following script first:
python balance_dataset.py
Then rename file as data.csv and run the following script:
python build_dataset.py

This script would add X.csv and Y.csv files to train, dev and test folders.
You are now ready to train the model.

3. Run the following script to start training - model versions are baseline, deeplayer, LSTMnet and convnet:
python train.py --model_dir experiments/baseline

4. Evaluate model on the test set, model versions are baseline, deeplayer, LSTMnet and convnet:
python evaluate.py --model_dir experiments/basemodel

5. Hyperparameter search:
python search_hyperparams.py --parent_dir experiments/learning_rate
