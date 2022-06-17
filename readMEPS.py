from collections import defaultdict
from copy import deepcopy
from sklearn import preprocessing
import math
import re
import random
import statistics
import sys
from itertools import count
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# filepath1 = r'./datasets/MEPS/' + "h181.csv"
# filepath2 = r'./datasets/MEPS/' + "h192.csv"
# df1 = pd.read_csv(filepath1)
# df2 = pd.read_csv(filepath2)
#
# # print(df1.head)
# # print("-----------------------------------------------------------------------\n\n")
# # print(df2.head)
# #
#
# # print("----------------------------------------------------------------------- \n\n")
# # print(df1.columns[957])
# # print(df1.columns[1142])
# # print(df1.columns[1094])
# # print(df1.columns[625])
# print("----------------------------------------------------------------------- \n\n")
# print(df2.columns[841])
# print(df2.columns[326])
# print(df2.columns[1133])
# print(df2.columns[1218])
# print(df2.columns[1265])
mtreatments = [1,2,3,5]
for m in mtreatments:
    enough = int(m * math.sqrt(800))
    print(enough)
