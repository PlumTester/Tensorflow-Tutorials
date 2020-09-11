# https://www.tensorflow.org/tutorials/estimator/boosted_trees_model_understanding#how_to_interpret_boosted_trees_models_both_locally_and_globally

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
sns_colors = sns.color_palette('colorblind')

from sklearn.metrics import roc_curve

from IPython.display import clear_output

import tensorflow as tf

# interpretting Boosted Tree models (locally / globally)
#   locally - understanding of model's predictions at individual example leve
#   globally - understanding of the model as a whole
# these understandings help detect bias and bugs during model development stage

# local interpretability
#   create and visualise per-instance contributions
#   distinguish from feature importances - refer to DFCs
# global interpretability
#   retrieve and visualise 
#       gain-based feature importances
#       permutation feature importances
#       show aggregated DFCs


def main():

    dftrain, dfeval, y_train, y_eval, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, feature_columns = load_data()

    num_examples = len(y_train)

    train_input_fn = make_input_fn(dftrain, y_train, num_examples)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_examples, num_epochs=1, shuffle=False)

    # estimator parameters
    #   center_bias must be True to get DFCs
    #   DFCs - directional feature contributions
    params = { \
        'n_trees' : 50, \
        'max_depth' : 3, \
        'n_batches_per_layer' : 1, \
        'center_bias' : True, \
        }

    # define estimator
    class_est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)

    # train
    class_est.train(train_input_fn, max_steps=100)

    # eval
    result = class_est.evaluate(eval_input_fn)
    class_result_series = pd.Series(data=list(result.values()), index=list(result.keys()), name='linear', dtype='float32')
    clear_output()
    
    # print(class_result_series)

    # Model interpretation and plotting
    # LOCAL INTERPRETABILITY
    pred_dicts = list(class_est.experimental_predict_with_explanations(eval_input_fn))

    # create DFC Pandas Dataframe
    labels = y_eval.values
    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
    df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
    
    # print(df_dfc.describe().transpose())

    # the sum of DFCs is the sum of the contributions + the bias is equal to the prediction for a given example
    # sum of DFCs + bias == probability
    bias = pred_dicts[0]['bias']
    dfc_prob = df_dfc.sum(axis=1) + bias
    np.testing.assert_almost_equal(dfc_prob.values, probs.values)

    # plot DFCs for an individual passenger (random 182 passenger)
    ID = 182
    example = df_dfc.iloc[ID]

    # view top 8 features
    TOP_N = 8
    
    # color code based on contributions' directionality
    # add feature values on figure
    ax = plot_example(example, dfeval, ID, TOP_N)
    ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
    ax.set_xlabel('Contribution to predicted probability', size=14)
    # plt.show()
    plt.close()

    # note on this chart - larger magnitude contributions have larger impact on model's prediction
    # negative contributions indicate feature value for this given example reduced the model's prediction
    # positive values contribute an increase in prediction

    # THIS IS A GOOD PLOT
    # plot example's DFCs compared with the entire distribution using a voilin plot
    dist_violin_plot(dfeval, df_dfc, ID, TOP_N)
    plt.title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
    # plt.show()
    plt.close()

    # GLOBAL INTERPRETABILITY
    #   gain-based feature importances: measure loss change when splitting on a particular feature
    #   permuatation importances: computed by evaluating model on evaluation set by shuffling each 
    #       feature 1 by 1 and attributing change in model performance to shuffled feature
    #   in general, permuation feature importance is preferred
    #   both methods can be unreliable IF
    #       predictor variavbles vary in scale of measurement
    #       predictor variables vary in number of categorie
    #       when features are correlated

    # in depth - http://explained.ai/rf-importance/index.html

    # # # GAIN-BASED feature importances - built into TF
    importances = class_est.experimental_feature_importances(normalize=True)
    df_imp = pd.Series(importances)

    # visualise importances
    N = 8
    ax = (df_imp.iloc[0:N][::-1].plot(kind='barh', color=sns_colors[0], title='Gain feature importances', figsize=(10,6)))
    ax.grid(False, axis='y')
    # plt.show()
    plt.close()

    # average absolute DFCs to understand impact at global level
    dfc_mean = df_dfc.abs().mean()
    N = 8

    # average and sort by absolute
    sorted_ix = dfc_mean.abs().sort_values()[-N:].index
    ax = dfc_mean[sorted_ix].plot(kind='barh', color=sns_colors[1], title='Mean | directional feature cotribution |', figsize=(10,6))
    ax.grid(False, axis='y')
    # plt.show()
    plt.close()

    # see how DFCs vary as featire value varies
    FEATURE = 'fare'
    feature = pd.Series(df_dfc[FEATURE].values, index=dfeval[FEATURE].values).sort_index()
    ax = sns.regplot(feature.index.values, feature.values, lowess=True)
    ax.set_ylabel('contribution')
    ax.set_xlabel(FEATURE)
    ax.set_xlim(0, 100)
    # plt.show()
    plt.close()


    # # # PERMUTATION feature importance
    features = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    importances = permutation_importances(class_est, dfeval, y_eval, accuracy_metric, features)

    def_imp = pd.Series(importances, index=features)

    sorted_ix = df_imp.abs().sort_values().index
    ax = df_imp[sorted_ix][-5:].plot(kind='barh', color=sns_colors[2], figsize=(10,6))
    ax.grid(False, axis='y')
    ax.set_title('Permutation feature importance')
    # plt.show()
    plt.close()

    return


def permutation_importances(est, X_eval, y_eval, metric, features):
    # column by column, shuffle values and observe effect on eval set

    # source - http://explained.ai/rf-importance/index.html
    # similar approach can be done during training if needed

    baseline = metric(est, X_eval, y_eval)
    imp = []

    for col in features:
        save = X_eval[col].copy()
        X_eval[col] = np.random.permutation(X_eval[col])
        m = metric(est, X_eval, y_eval)
        X_eval[col] = save
        imp.append(baseline - m)

    return np.array(imp)

def accuracy_metric(est, X, y):

    eval_input_fn = make_input_fn(X, y, len(y), num_epochs=1, shuffle=False)
    
    return est.evaluate(input_fn=eval_input_fn)['accuracy']

# plotting BOILERPLATE code - color coded based DFCs plot
def _get_color(value):
    # make positive DFCs plot green, negative DFCs plot red
    green, red = sns.color_palette()[2:4]
    if value >= 0: 
        return green
    else:
        return red



def _add_feature_values(feature_values, ax):
    # display feature's values on left of plot
    x_coord = ax.get_xlim()[0]
    OFFSET = 0.15

    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))

    from matplotlib.font_manager import FontProperties

    font = FontProperties()
    font.set_weight('bold')

    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue')

    return



def plot_example(example, dfeval, ID, TOP_N):
    sorted_ix = example.abs().sort_values()[-TOP_N:].index
    example = example[sorted_ix]
    colors = example.map(_get_color).tolist()
    
    ax = example.to_frame().plot(kind='barh', color=[colors], legend=None, alpha=0.75, figsize=(10,6))
    ax.grid(False, axis='y')
    ax.set_yticklabels(ax.get_yticklabels(), size=14)

    # add feature values
    _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)
    return ax


# Boilerplate plotting code - violin plot
def dist_violin_plot(dfeval, df_dfc, ID, TOP_N):

    # Initialize plot.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create example dataframe.
    example = df_dfc.iloc[ID]
    sorted_ix = example.abs().sort_values()[-TOP_N:].index
    example = example[sorted_ix]
    example_df = example.to_frame(name='dfc')

    # Add contributions of entire distribution.
    parts=ax.violinplot([df_dfc[w] for w in sorted_ix],
                    vert=False,
                    showextrema=False,
                    widths=0.7,
                    positions=np.arange(len(sorted_ix)))
    face_color = sns_colors[0]
    alpha = 0.15
    for pc in parts['bodies']:
        pc.set_facecolor(face_color)
        pc.set_alpha(alpha)

    # Add feature values.
    _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)

    # Add local contributions.
    ax.scatter(example,
                np.arange(example.shape[0]),
                color=sns.color_palette()[2],
                s=100,
                marker="s",
                label='contributions for example')

    # Legend
    # Proxy plot, to show violinplot dist on legend.
    ax.plot([0,0], [1,1], label='eval set contributions\ndistributions',
            color=face_color, alpha=alpha, linewidth=10)
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large',
                        frameon=True)
    legend.get_frame().set_facecolor('white')

    # Format plot.
    ax.set_yticks(np.arange(example.shape[0]))
    ax.set_yticklabels(example.index)
    ax.grid(False, axis='y')
    ax.set_xlabel('Contribution to predicted probability', size=14)

    return




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



def load_data():

    # load titanic data
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')

    # # explore the data
    # print(dftrain.head())
    # print()
    # print(dftrain.describe())
    # print()

    # # 627 passengers in train
    # # 264 passengers in evaluation
    # print(dftrain.shape[0], dfeval.shape[0])

    
    dftrain.age.hist(bins=20)
    # plt.show()
    plt.close()

    # approx 2x male passengers vs female passengers
    dftrain.sex.value_counts().plot(kind='barh')
    # plt.show()
    plt.close()

    # major in third class
    dftrain['class'].value_counts().plot(kind='barh')
    # plt.show()
    plt.close()

    # females have much higher chance of surviving versus males - predictive feature for the model
    pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
    # plt.show()
    plt.close()

    # FEATURE ENGINEERING FOR MODEL
    CATEGORICAL_COLUMNS = ['sex', \
                            'n_siblings_spouses', \
                            'parch', \
                            'class', \
                            'deck', \
                            'embark_town', \
                            'alone']

    NUMERIC_COLUMNS = ['age', \
                            'fare']
    
    def one_hot_cat_column(feature_name, vocab):
        return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))
        
    feature_columns = []

    for feature_name in CATEGORICAL_COLUMNS:
        # get a set of range of feature values
        vocabulary = dftrain[feature_name].unique()
        # define range of categorical values for feature
        feature_columns.append(one_hot_cat_column(feature_name, vocabulary))


    for feature_name in NUMERIC_COLUMNS:
        # define numerical dtype
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


    # # see an example of what one_hot_cat_column does
    # # tuple of numerical dummy variables basically
    # example = dict(dftrain.head(1))
    # class_feature_column = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
    # print('Feature value: "{}"'.format(example['class'].iloc[0]))
    # print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_feature_column])(example).numpy())

    # # can also view all feature column transformations together
    # print(tf.keras.layers.DenseFeatures(feature_columns)(example).numpy())

    return dftrain, dfeval, y_train, y_eval, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, feature_columns




if __name__ == "__main__":
    main()