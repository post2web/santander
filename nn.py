import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc
from sklearn.cross_validation import train_test_split
import math
import argparse
from pprint import pprint
import shutil
import pickle, os

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--inputs', type=str, required=True)
    parser.add_argument('--log_targets', action='store_true')
    parser.add_argument('--fresh_start', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--l2_regularizer', type=float, default=0.0001)
    parser.add_argument('--hidden_units', type=str, required=True)
    parser.add_argument('--max_not_improve', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=10000000)
    
    params = vars(parser.parse_args())
    
    params['hidden_units'] = [int(c.strip()) for c in params['hidden_units'].split(',')]
    params['model_dir'] = 'nn_model_dirs/' + params['name'] + '/'
    if params['inputs'][-1] != '/':
        params['inputs'] += '/'

    return params

def get_input_fns(params):
    data = pd.read_hdf(params['inputs'] + 'train.h5', 'data')

    train, valid = train_test_split(data, test_size=.1, random_state=0)
    
    print(len(train), len(valid), len(data))
    
    X_train = train
    y_train = X_train['target'].copy()
    del X_train['target']

    X_valid = valid
    y_valid = X_valid['target'].copy()
    del X_valid['target']

    if params['log_targets']:
        y_train = np.log(y_train)
        y_valid = np.log(y_valid)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'inputs':X_train.values},
        y=y_train,
        batch_size=params['batch_size'],
        num_epochs=None,
        shuffle=True
    )
    valid_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'inputs':X_valid.values},
        y=y_valid,
        batch_size=128,
        num_epochs=1,
        shuffle=False
    )
    
    feature_columns = [
        fc.numeric_column('inputs', shape=X_train.shape[1])
    ]
        
    return train_input_fn, valid_input_fn, feature_columns

if __name__ == '__main__':
    
    params = parse_args()
    pprint(params)
    
    if 'fresh_start' in params and params['fresh_start']:
        shutil.rmtree(params['model_dir'], ignore_errors=True)

    # remove the expoted results
    shutil.rmtree(params['model_dir'] + 'result/', ignore_errors=True)
    os.makedirs(params['model_dir'],exist_ok=True)
    pickle.dump(params, open(params['model_dir'] + "params.p", "wb" ) )
    
    train_input_fn, valid_input_fn, feature_columns = get_input_fns(params)

    config = tf.estimator.RunConfig(
        keep_checkpoint_max=params['max_not_improve']+1,
    )
    
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params['learning_rate'])
    
    estimator = tf.estimator.DNNRegressor(
        hidden_units=params['hidden_units'],
        feature_columns=feature_columns,
        model_dir=params['model_dir'],
        dropout=params['dropout_rate'],
        optimizer=optimizer,
        config=config
    )
    
    tf.logging.set_verbosity(tf.logging.ERROR)

    best_loss = np.inf
    best_global_step = 0
    not_improve_counter = 0

    while not_improve_counter < params['max_not_improve']:

        estimator.train(input_fn=train_input_fn, steps=500)
        
        eval_result = estimator.evaluate(input_fn=valid_input_fn)
        print(eval_result)
        if eval_result['loss'] < best_loss:
            best_loss = eval_result['loss']
            not_improve_counter = 0
        else:
            not_improve_counter += 1

        if params['max_steps'] <= eval_result['global_step']:
            print('max_steps reached')
            break
            
        
    

