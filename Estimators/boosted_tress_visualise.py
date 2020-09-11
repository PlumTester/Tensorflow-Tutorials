# https://www.tensorflow.org/tutorials/estimator/boosted_trees_model_understanding#average_absolute_dfcs

import numpy as np
import pandas as pd

from numpy.random import uniform, seed
from scipy.interpolate import griddata


from matplotlib import pyplot as plt
import seaborn as sns

sns_colors = sns.color_palette('colorblind')

from sklearn.metrics import roc_curve

from IPython.display import clear_output

import tensorflow as tf




def main():

    # create fake data
    # based on formula z = x * e^(-x^2 - y^2)
    seed(0)
    npts = 5000
    x = uniform(-2, 2, npts)
    y = uniform(-2, 2, npts)
    z = x * np.exp(-x**2 - y**2)



    xy = np.zeros((2, np.size(x)))
    xy[0] = x
    xy[1] = y

    xy = xy.transpose()

    # print('x: {}'.format(x))
    # print('y: {}'.format(y))
    # print('z: {}'.format(z))
    # print(xy)

    # prep data for training
    df = pd.DataFrame({'x' : x, 'y' : y, 'z' : z})

    xi = np.linspace(-2.0, 2.0, 200)
    yi = np.linspace(-2.1, 2.1, 210)
    xi, yi = np.meshgrid(xi, yi)

    df_predict = pd.DataFrame({ 'x' : xi.flatten(), 'y' : yi.flatten()})
    predict_shape = xi.shape

    # print(predict_shape)

    # test_contour_plot(xy, z, xi, yi, df)

    # prepare model
    feature_columns = [tf.feature_column.numeric_column('x'), tf.feature_column.numeric_column('y')]

    num_examples = len(df.z)

    # linear model
    train_input_fn = make_input_fn(df, df.z, num_examples)
    lin_est = tf.estimator.LinearRegressor(feature_columns)
    lin_est.train(train_input_fn, max_steps=500)
    clear_output()
    
    # straight lines means not a very good fit
    plot_contour(xi, yi, predict(lin_est, df_predict, predict_shape))
    # plt.show()
    plt.close()

    # lets try to fit GBDT model to understand how the model fits the function
    n_trees = 37

    bt_est = tf.estimator.BoostedTreesRegressor(feature_columns, n_batches_per_layer=1, n_trees=n_trees)
    bt_est.train(train_input_fn, max_steps=500)
    clear_output()

    plot_contour(xi, yi, predict(bt_est, df_predict, predict_shape))
    plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20)
    plt.show()
    plt.close()

    return


def predict(est, df_predict, predict_shape):
    # predictions from a given estimator
    predict_input_fn = lambda: tf.data.Dataset.from_tensors(dict(df_predict))

    preds = np.array([p['predictions'][0] for p in est.predict(predict_input_fn)])

    return preds.reshape(predict_shape)


def make_input_fn(data_df, label_df, num_examples, num_epochs=None, shuffle=True):

    def input_function():
        # note instantiate TF Dataset from pandas df by dict transformation
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(num_examples)

        # for training - cycle through dataset as many times as needed (n_epochs=None)
        ds = ds.repeat(num_epochs)

        # in memory, training doesn't use batching
        ds = ds.batch(num_examples)

        return ds

    return input_function


def test_contour_plot(xy, z, xi, yi, df):
    
    zi = griddata(xy, z, (xi, yi), method='linear', fill_value='0')
    plot_contour(xi, yi, zi)

    plt.scatter(df.x, df.y, marker='.')
    plt.title('Contour on training data')

    plt.show()
    plt.close()

    return



def plot_contour(x, y, z, **kwargs):
    # Grid the data.
    plt.figure(figsize=(10, 8))
    # Contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(x, y, z, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(x, y, z, 15, vmax=abs(z).max(), vmin=-abs(z).max(), cmap='RdBu_r')

    # Draw colorbar.
    plt.colorbar()

    # Plot data points.
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    return



if __name__ == "__main__":
    main()