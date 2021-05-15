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


def create_gr_feats(data):
    """

    :param data:
    :return:
    """
    for col in ['Walls', 'District', 'Okrug']:
        for n, f in [('mean', np.mean), ('min', np.nanmin), ('max', np.nanmax)]:
            data[n + '_size_by_' + col] = data.groupby(col)['Size'].transform(f)
    for col in ['Size', 'Room', 'Balcony', 'Floor', 'FloorsTotal',
               'Walls','Age','Lift', 'District', 'Okrug']:
        data[col + '_cnt'] = data[col].map(data[col].value_counts(dropna = False))
