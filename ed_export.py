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
    parser.add_argument('--folder', type=str, required=True)
    folder = vars(parser.parse_args())['folder']
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
    train = pd.read_hdf('data/train_loged.h5', 'data')
    test = pd.read_hdf('data/test_loged.h5', 'data')
    
    train_targets = train['target'].copy()
    del train['target']
    
    features = pd.read_hdf('data/importance.h5', 'data').iloc[:params['n_features']]
    train = train[features.index]
    test = test[features.index]
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'inputs':train.values},
        batch_size=500,
        num_epochs=1,
        shuffle=False
    )
    result = [(train_input_fn, train.index, train_targets)]
    
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'inputs':test.values},
        batch_size=500,
        num_epochs=1,
        shuffle=False
    )
    result.append((test_input_fn, test.index))
    
    return result

def predict(key, input_fn):
        
    result = pd.DataFrame()
    preds = estimator.predict(
        input_fn=input_fn,
        checkpoint_path=params['model_dir']+'model.ckpt-%d' % params['best_step']
    )
    results = []
    for pred in preds:
        results.append(pred[key])
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    
    params = parse_args()
    pprint(params)
    
    best_step = get_best_step(params)
    params['best_step'] = best_step
    print('best_step', best_step)
    
    (train, train_index, y), (test, test_index) = get_data(params)

    estimator = tf.estimator.Estimator(
        model_dir=params['model_dir'],
        model_fn=model_fn,
        params=params
    )
    
    result_dir = params['model_dir'] + 'result/'
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir)
    
    results = predict('Z', input_fn=train)
    results.index = train_index
    results['target'] = y.values
    results.to_hdf(params['export_name']+'train.h5', 'data')
    
    results = predict('reconstruction', input_fn=train)
    results.to_hdf(params['export_name']+'train_reconstruction.h5', 'data')
    
    results = predict('Z', input_fn=test)
    results.index = test_index
    results.to_hdf(params['export_name']+'test.h5', 'data')
