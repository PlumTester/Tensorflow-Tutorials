# https://www.tensorflow.org/guide/checkpoint

import tensorflow as tf
import tensorflow.compat.v1 as tf_compat


class Net(tf.keras.Model):
    # simple linear model

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)



def main():

    # saving_from_keras_training_APIs()

    # writing_checkpoints_manually()

    # restore_and_continue_training()

    write_ckpts_with_estimator()

    load_estimator_ckpt()

    return



# Saving from tf.keras training APIs
#   tf.keras.Model.save_weights saves a TensorFlowcheckpoint
def saving_from_keras_training_APIs():
    net = Net()
    net.save_weights('easy_checkpoint')

    return


# define toy dataset
def toy_dataset():
    
    inputs = tf.range(10.)[:, None]
    # print(inputs)
    
    labels=inputs * 5. + tf.range(5.)[None, :]
    # print(labels)

    return tf.data.Dataset.from_tensor_slices( dict(x=inputs, y=labels) ).repeat().batch(2)


# define optimization step
def train_step(net, example, optimizer):
    # Trains 'net' on 'example' using 'optimizer'

    with tf.GradientTape() as tape:
        output = net(example['x'])
        loss = tf.reduce_mean(tf.abs(output - example['y']))

    variables = net .trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


# Writing Checkpoints
#   persistent TF model state is stored in tf.Variable objects
#   created via high-level APIs like tf.keras.layers or tf.keras.Model
#   attach variables to Python objects and reference them
#   example constructs simple linear model, writes checkpoints for all variables
def writing_checkpoints_manually():

    net = Net()
    
    # need to make tf.train.Checkpoint object - objects you want to checkpoint are set as attributes on the object
    opt = tf.keras.optimizers.Adam(0.1)
    dataset = toy_dataset()
    iterator = iter(dataset)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    # ^NOTE - ./ instead of / -- without the . it will raise a PermissionDeniedError

    train_and_checkpoint(net, manager, ckpt, iterator, opt)

# define a training loop
#   creates instance of model and optimizer
#   gathers them into a  tf.trian.Checkpoint object
#   calls training step in loop on aech batch of data
#   periodically writes checkpoints to disk
def train_and_checkpoint(net, manager, ckpt, iterator, opt):

    ckpt.restore(manager.latest_checkpoint)
    # ckpt.restore(manager.latest_checkpoint).expect_partial()
    # NOTE expect partial will avoid warning should avoid the unresolved object warning - didn't work

    if manager.latest_checkpoint:
        print("restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for _ in range(50):
        example = next(iterator)
        loss = train_step(net, example, opt)
        ckpt.step.assign_add(1)

        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {} : {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(loss.numpy()))

    return

# after you write checkpoints manually via train_and_checkpoint
# can pass a new model and manager, but conitnue training exactly where it was left off
def restore_and_continue_training():

    opt = tf.keras.optimizers.Adam(0.1)
    net = Net()
    dataset = toy_dataset()
    iterator = iter(dataset)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    train_and_checkpoint(net, manager, ckpt, iterator, opt)

    # NOTE - tf.train.CheckpointManager deletes old checkpoints
    # configured to keep the three most recent checkpoints
    print(manager.checkpoints)

    # output: ['./tf_ckpts\\ckpt-13', './tf_ckpts\\ckpt-14', './tf_ckpts\\ckpt-15']
    # these paths are prefixed for an index file and one or more data files which contain the variable values
    # these prefixes are grouped in a checkpoint file './tf_ckpts/checkpoint' where CheckpointManager saves its state

    # NOTE unresolved object warnings is just because training is not continued - https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-objec
    

    return


# # Saving object-based checkpoints with Estimator
# Estimators save ckpts with variable names rather than object graph
# accepted name-based checkpoints - variable names may change when moving parts of model outside of Estimatpor's model_fn
# saving object-based checkpoints make it easier to train model inside an estimator and then use it outside of one

def write_ckpts_with_estimator():
    
    tf.keras.backend.clear_session()
    est = tf.estimator.Estimator(model_fn, './tf_estimator_example/')
    est.train(toy_dataset, steps=10)

    return 

def load_estimator_ckpt():

    opt = tf.keras.optimizers.Adam(0.1)
    net = Net()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=opt, net=net)

    ckpt.restore(tf.train.latest_checkpoint('./tf_estimator_example/'))
    print(ckpt.step.numpy()) # from est.train(..., steps=10)

    return



def model_fn(features, labels, mode):

    net = Net()
    opt = tf.keras.optimizers.Adam(0.1)
    ckpt = tf.train.Checkpoint(step=tf_compat.train.get_global_step(), optimizer=opt, net=net)

    with tf.GradientTape() as tape:
        output = net(features['x'])
        loss = tf.reduce_mean(tf.abs(output - features['y']))
    
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)

    return tf.estimator.EstimatorSpec(mode, \
                                        loss=loss, \
                                        train_op=tf.group(opt.apply_gradients(zip(gradients, variables)), \
                                        ckpt.step.assign_add(1)), \
                                        scaffold=tf_compat.train.Scaffold(saver=ckpt))


if __name__ == "__main__":
    main()