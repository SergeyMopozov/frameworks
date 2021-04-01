'''
File containe routine procedur for explore data analisys
'''

#import pandas as pd

def show_info(data):
    '''

    :param data: pandas data frame
    :return:
    '''
    print(data.head())
    print(data.info())
    print(data.describe())


