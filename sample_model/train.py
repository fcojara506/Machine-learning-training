import pandas as pd
from sklearn import metrics
import joblib

import os
import argparse

import config
import model_dispatcher

def run_model(fold,model):
    # read the training data
    df = pd.read_csv(config.TRAINING_FILE)
    
    #separate the model based on the fold
    #training set
    df_train = df[df.kfold != fold].reset_index(drop = True)
    
    #validation set
    df_valid = df[df.kfold == fold].reset_index(drop = True)
    
    # convert to numpy to use in the model
    #training
    x_train = df_train.drop('label', axis = 1).values
    y_train = df_train['label'].values
    
    #validation
    x_valid = df_valid.drop('label', axis = 1).values
    y_valid = df_valid['label'].values
    
    
    #create a simple model from model dispatcher
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    
    #predict the hold-out fold
    pred = clf.predict(x_valid)
    
    #metrics
    accuracy = metrics.accuracy_score(y_valid, pred)
    print(f'Fold={fold}, Model = {model}, Accuracy = {accuracy}')
    
    # save the model
    joblib.dump(clf, filename = os.path.join(config.MODEL_OUTPUT, f'dt_{model}_{fold}.bin'))

if __name__=='__main__':
    #initialise parser 
    parser = argparse.ArgumentParser()
    
    #add argument 'fold
    parser.add_argument('--fold', type = int, default=0)
    #add argument 'model'
    parser.add_argument('--model', type = str, default = 'decision_tree_gini')
    
    args = parser.parse_args()
    
    # run final model
    run_model(
        fold=args.fold,
        model = args.model
        )
    



