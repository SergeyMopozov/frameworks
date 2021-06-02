
import matplotlib.pyplot as plt
import numpy as np


def split_hist(data, bins=20, percentile=75):
    """

    :param data:
    :param percentile:
    :return:
    """
    fst_half = data[data < np.percentile(data, percentile)]
    snd_half = data[data >= np.percentile(data, percentile)]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].hist(fst_half, bins=bins)
    axs[0].grid()
    axs[0].set_title(f'<{percentile} percentile')

    axs[1].hist(snd_half, bins=bins)
    axs[1].grid()
    axs[1].set_title(f'>={percentile} percentile')
    plt.show()
