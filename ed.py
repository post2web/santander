import argparse
from pprint import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import shutil, os
import pickle

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--log_input', action='store_true')
    parser.add_argument('--fresh_start', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--noise_level', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--l2_regularizer', type=float, default=0.0000000001)
    parser.add_argument('--l1_regularizer', type=float, default=0.0000000001)
    parser.add_argument('--hidden_units', type=str, required=True)
    parser.add_argument('--max_not_improve', type=int, default=25)
    parser.add_argument('--max_steps', type=int, default=3000000)
    parser.add_argument('--n_features', type=int, default=1000)
    
    params = vars(parser.parse_args())
    params['hidden_units'] = [int(c.strip()) for c in params['hidden_units'].split(',')]
    
    base = './'
    params['model_dir'] = base+'ed_model_dirs/' + params['name'] + '/'
    params['export_name'] = params['model_dir'] + 'result'

    return params

def get_data(params):
    """
    data = pd.read_hdf('data/train_loged.h5', 'data').append(
        pd.read_hdf('data/test_loged.h5', 'data')
    )
    del data['target']
    features = pd.read_hdf('data/importance.h5', 'data').iloc[:params['n_features']]
    data = data[features.index]
    
    assert data.shape[1] == params['n_features']
    assert not np.any(np.isnan(data))
    assert np.all(np.isfinite(data))
    
    train, valid = train_test_split(data, test_size=.1, random_state=0)
    """
    valid = pd.read_hdf('data/train_loged.h5', 'data')
    del valid['target']
    train = pd.read_hdf('data/test_loged.h5', 'data')
    features = pd.read_hdf('data/importance.h5', 'data').iloc[:params['n_features']]
    train = train[features.index]
    valid = valid[features.index]
    
    return train, valid


def inputs(train, valid):
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'inputs':train.values},
        batch_size=params['batch_size'],
        num_epochs=None,
        num_threads=4,
        shuffle=True
    )
    valid_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'inputs':valid.values},
        batch_size=params['batch_size'],
        num_epochs=1,
        shuffle=False
    )

    return train_input_fn, valid_input_fn


def model_fn(features, labels, mode, params):

    default_params = {
        'noise_level': 0,
        'learning_rate':.001,
        'dropout_rate':0.0,
        'l2_regularizer':0.0,
    }
    default_params.update(params)
    params = default_params
    
    encoder_hidden_units = params['hidden_units'][:]
    
    z_units = encoder_hidden_units.pop()
    
    decoder_hidden_units = encoder_hidden_units[:-1]
    decoder_hidden_units.reverse()
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    input_layer = features['inputs']
    
    input_size = input_layer.get_shape().as_list()[1]
    
    #net = tf.layers.batch_normalization(input_layer, training=is_training)
    #net = tf.contrib.layers.layer_norm(input_layer)
    
    net = input_layer
    
    if params['noise_level'] and is_training:
        noise = tf.random_normal(tf.shape(net), stddev=params['noise_level'])
        net = tf.nn.relu(net + noise)


    regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=params['l1_regularizer'],
        scale_l2=params['l2_regularizer']
    )
    
    for i, units in enumerate(encoder_hidden_units):
        net = tf.layers.dense(
            net, units,
            #kernel_initializer=tf.truncated_normal_initializer(stddev=0.5),
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)
        net = tf.layers.dropout(net, rate=params['dropout_rate'], training=is_training)
        
    Z = tf.layers.dense(
        net, z_units, activation=None)
    
    for units in decoder_hidden_units:
        net = tf.layers.dense(
            net, units,
            activation=tf.nn.relu,
            #kernel_initializer=tf.truncated_normal_initializer(stddev=0.5),
            kernel_regularizer=regularizer
        )
        
        #net = tf.layers.dropout(net, rate=params['dropout_rate'], training=is_training)

    reconstruction = tf.layers.dense(net, units=input_size, activation=None)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={
            'Z':Z,
            'reconstruction':reconstruction
        })

    loss = tf.losses.mean_squared_error(input_layer, reconstruction)
    reg = tf.losses.get_regularization_loss()
    loss = loss + reg
    
    tf.summary.histogram('Z', Z)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('reg', reg)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
    
#    gvs = optimizer.compute_gradients(loss)
#    capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
#    train_op = optimizer.apply_gradients(
#        capped_gvs, global_step=tf.train.get_global_step())

    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)
    return estimator_spec


if __name__ == '__main__':
    
    params = parse_args()
    pprint(params)
    tf.logging.set_verbosity(tf.logging.WARN)
    
    if 'fresh_start' in params and params['fresh_start']:
        shutil.rmtree(params['model_dir'], ignore_errors=True)

    # remove the expoted results
    shutil.rmtree(params['model_dir'] + 'result/', ignore_errors=True)
    
    os.makedirs(params['model_dir'],exist_ok=True)
    pickle.dump(params, open(params['model_dir'] + "params.p", "wb" ) )
    
    train, valid = get_data(params)
    train_input_fn, valid_input_fn = inputs(train, valid)
    
    config = tf.estimator.RunConfig(
        keep_checkpoint_max=params['max_not_improve']+1,
    )

    estimator = tf.estimator.Estimator(
        model_dir=params['model_dir'],
        model_fn=model_fn,
        params=params,
        config=config
    )
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    best_loss = np.inf
    not_improve_counter = 0

    while not_improve_counter < params['max_not_improve']:

        estimator.train(
            input_fn=train_input_fn,
            steps=500)

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
