import numpy as np

def imputer_for_nans(data, group_col_name, imputer_col_name, statistics='median'):
    """
    Fill NA values by statistic value in frame
    :param data:
    :param group_col_name:
    :param imputer_col_name:
    :param statistics:
    :return:
    """
    return data[imputer_col_name].fillna(data.groupby(group_col_name)[imputer_col_name].transform(statistics))


def create_col_with_min_freq(data, col, min_freq=10):
    """
    replace rare values (less than min_freq rows) in feature by RARE_VALUE
    :param data:
    :param col:
    :param min_freq:
    :return:
    """

    data[col + '_fixed'] = data[col].astype(str)
    data.loc[data[col + '_fixed'].value_counts()[data[col + '_fixed']].values < min_freq, col + '_fixed'] = "RARE_VALUE"
    data.replace({'nan': np.nan}, inplace=True)


def create_gr_feats(data, cat_feats, num_feats, min_freq):
    """
    create aggregation feats for numeric features based on categorical ones and count its
    :param num_feats:
    :param cat_feats:
    :param min_freq:
    :param data:
    :return:
    """
    # create aggregation feats for numeric features based on categorical ones
    for cat_col in cat_feats:
        create_col_with_min_freq(data, cat_col, min_freq)
        for num_col in num_feats:
            for n, f in [('mean', np.mean), ('min', np.nanmin), ('max', np.nanmax), ('std', np.nanstd)]:
                data['FIXED_' + n + '_' + num_col + '_by_' + cat_col] = \
                    data.groupby(cat_col + '_fixed')[num_col].transform(f)

    # create features with counts
    for col in cat_feats:
        data[col + '_cnt'] = data[col].map(data[col].value_counts(dropna=False))

