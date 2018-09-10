import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

train_data = pd.read_csv('data/train.csv')
digit_num = train_data['label'].unique().shape[0]
pd.set_option('display.max_rows', 20)


data_0 = train_data[train_data['label'] == 0].iloc[:100, 1:].values
data_1 = train_data[train_data['label'] == 1].iloc[:100, 1:].values


def get_L2_dist_m2m(Xs, Ys):
    """
      X [3,4],     Y [0,0],
        [5,12]       [1,1]

      X^2 [9, 16],   sum(axis=1) [25,  25]     [X1, X1,
          [25,144]                169, 169]     X2, X2]

      Y^2 [0, 0],   sum(axis=1)   [0, 2]       [Y1, Y2],
          [1, 1]                   0, 2]        Y1, Y2 

      2(X dot Y.T)                [0 14],      [X1Y1, X1Y2],
                                  [0 34]       [X2Y1, X2Y2]

      result     [25+0-0=25,    25+2-14=13],   [X1Y1, X1Y2], 
                 [169+0-0=169, 169+2-34=137]   [X2Y1, X2Y2]             
    """
    sum_X_sqr = np.square(Xs).sum(axis = 1)
    matrix_X_sqr = np.tile(sum_X_sqr, (sum_X_sqr.shape[0], 1)).T
    
    sum_Y_sqr = np.square(Ys).sum(axis = 1)
    matrix_Y_sqr = np.tile(sum_Y_sqr, (sum_Y_sqr.shape[0], 1))
    
    return np.sqrt(matrix_X_sqr - 2 * np.dot(Xs, Ys.T) + matrix_Y_sqr)


"""
  Function TEST
"""
# result = get_L2_dist_m2m(np.array([[3,4],[5,12]]),np.array([[0,0],[1,1]]))
# print(result)

_0to0 = get_L2_dist_m2m(data_0, data_0)
# _0to1 = get_L2_dist_m2m(data_0, data_1)
# _1to0 = get_L2_dist_m2m(data_1, data_0)
# _1to1 = get_L2_dist_m2m(data_1, data_1)

print(_0to0)
# display(_0to1)


