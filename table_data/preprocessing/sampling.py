import pandas as pd
from sklearn.utils import shuffle


def upsampling(X_train, y_train, repeat):
    """

    :param X_train:
    :param y_train:
    :param repeat:
    :return:
    """
    features_zeros = X_train[y_train == 0]
    features_ones = X_train[y_train == 1]
    target_zeros = y_train[y_train == 0]
    target_ones = y_train[y_train == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    return shuffle(features_upsampled, target_upsampled)


def downsample(X_train, y_train, fraction):
    """

    :param X_train:
    :param y_train:
    :param fraction:
    :return:
    """
    features_zeros = X_train[y_train == 0]
    features_ones = X_train[y_train == 1]
    target_zeros = y_train[y_train == 0]
    target_ones = y_train[y_train == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)

    return features_downsampled, target_downsampled