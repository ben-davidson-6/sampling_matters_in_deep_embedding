from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


from pprint import pprint
import tensorflow.estimator as estimator
import logging
import os

from models import MNISTModel, Mnist

logging.getLogger().setLevel(logging.INFO)

# define parameters used
epochs_to_train = 20
val_examples = 10000
train_examples = 40000
batch_size = 256
throttle_mins = 0
params = dict()
params['batch_size'] = batch_size
params['steps_per_epoch'] = train_examples // params['batch_size']
params['total_steps_train'] = params['steps_per_epoch'] * epochs_to_train
params['throttle_eval'] = throttle_mins * 60
params['momentum'] = 0.9
params['initial_lr'] = 0.1
params['alpha'] = 0.2
params['nu'] = 0.
params['cutoff'] = 0.5
params['add_summary'] = True
params['beta_0'] = 1.2
model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist_model')

pprint(params)
print('placing model artifacts in {}'.format(model_dir))

# define model and data
model = MNISTModel()
mnist_data = Mnist(params['batch_size'])

run_config = estimator.RunConfig(
    save_checkpoints_steps=params['steps_per_epoch'],
    save_summary_steps=200,
    keep_checkpoint_max=10)

mnist_estimator = estimator.Estimator(
    model_dir=model_dir,
    model_fn=model.model_fn,
    params=params,
    config=run_config)

# training/evaluation specs for run
train_spec = estimator.TrainSpec(
    input_fn=mnist_data.build_training_data,
    max_steps=params['total_steps_train'],)

eval_spec = estimator.EvalSpec(
    input_fn=mnist_data.build_validation_data,
    steps=None,
    throttle_secs=params['throttle_eval'],
    start_delay_secs=0)

# run train and evaluate
estimator.train_and_evaluate(
    mnist_estimator,
    train_spec,
    eval_spec)
