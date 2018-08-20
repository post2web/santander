from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
from pprint import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from ed import model_fn
from ed import get_data as ed_get_data
import os, shutil, sys
from sklearn.cross_validation import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--inputs', type=str, required=True)
    folder = vars(parser.parse_args())['model']
    if folder[-1] != '/':
        folder += '/'
    
    params = pickle.load(open(folder+'params.p', 'rb'))
    params['export_name'] = params['model_dir'] + 'result/'
    
    if params['model_dir'][-1] != '/':
        params['model_dir'] = params['model_dir'] + '/'

    return params

def get_best_step(params):
    event_acc = EventAccumulator(params['model_dir'] + 'eval')
    event_acc.Reload()
    loss = pd.DataFrame(event_acc.Scalars('loss'))
    return loss.sort_values('value')['step'].iloc[0]

def get_data(params):
    
    data = pd.read_hdf(params['inputs'] + 'train.h5', 'data')

    _, valid = train_test_split(data, test_size=.1, random_state=0)
    
    test = pd.read_hdf(params['inputs'] + 'test.h5', 'data')

    X_valid = valid
    y_valid = X_valid['target'].copy()
    del X_valid['target']
    
    
    valid_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'inputs':X_valid.values},
        batch_size=params['batch_size'],
        num_epochs=1,
        shuffle=False
    )
    
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'inputs':test.values},
        batch_size=params['batch_size'],
        num_epochs=1,
        shuffle=False
    )
    
    feature_columns = [
        tf.feature_column.numeric_column('inputs', shape=X_valid.shape[1])
    ]

    return (valid_input_fn, y_valid), (test_input_fn, test.index), feature_columns

def predict(input_fn):
        
    result = pd.DataFrame()
    preds = estimator.predict(
        input_fn=input_fn,
        checkpoint_path=params['model_dir']+'model.ckpt-%d' % params['best_step']
    )
    results = []
    for pred in preds:
        results.append(pred['predictions'][0])
    
    return pd.DataFrame(results, columns=['target'])

if __name__ == '__main__':
    
    params = parse_args()
    pprint(params)
    
    best_step = get_best_step(params)
    params['best_step'] = best_step
    print('best_step', best_step)
    
    (valid_input_fn, y_valid), (test_input_fn, test_index), feature_columns = get_data(params)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=params['learning_rate'])
    
    estimator = tf.estimator.DNNRegressor(
        hidden_units=params['hidden_units'],
        feature_columns=feature_columns,
        model_dir=params['model_dir'],
        dropout=params['dropout_rate'],
        optimizer=optimizer
    )
    
    shutil.rmtree(params['export_name'], ignore_errors=True)
    os.makedirs(params['export_name'])
    
    results = predict(valid_input_fn)
    results['y_true'] = y_valid.values
    results.to_hdf(params['export_name']+'valid.h5', 'data')
    
    results = predict(test_input_fn)
    results['ID'] = test_index
    results = results[['ID', 'target']]
    results.to_csv(params['export_name']+'submission.csv', index=False)
