
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

from columns import Table, Col, Sym, Num, TreeNode
# from measure import measure_final_score,calculate_recall,calculate_precision,calculate_accuracy
# from metrics import getMetrics

from adversarial_models import *
from utils import *

import lime
import lime.lime_tabular

# ------------------------------------------------------------------------------
# TreeNode Class Helper Fuctions: Functional Tree Traversal
# ------------------------------------------------------------------------------

def nodes(root):  # gets all the leaf nodes
    if root:
        for node in nodes(root.leftNode): yield node  # yield returns from a function without destroying it
        if root.leaf:  yield root
        for node in nodes(root.rightNode): yield node


def names(root: TreeNode):  # gets all the col names of the node
    for node in nodes(root):
        for i in range(len(node.leftTable.cols) - 1):
            print(node.leftTable.cols[i].name)


def rowSize(t): return len(t.leftTable.rows)  # gets the size of the rows


def leafmedians(root, how=None):  # for all of the leaves from smallest to largest print len of rows & median
    MedianTable = Table(222)
    MedianTable.create_cols(root.header)
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        mid = [col.mid() for col in t.cols]
        MedianTable + mid
        # print(len(t.rows), [col.mid() for col in t.cols], t.cols[-1].count)
    MedianTable.encode_lines()
    return MedianTable


def getLeafData(root, samples_per_leaf,
                how=None):  # for all of the leaves from smallest to largest print len of rows & median
    EDT = Table(samples_per_leaf)
    EDT.create_cols(root.header)
    counter = 0
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        for i in range(samples_per_leaf):
            randomrow = random.choice(t.rows)
            EDT + randomrow
            counter += 1
    EDT.encode_lines()
    return EDT


def isValid(self, row):
    for val in row:
        if val == '?':
            return False
    return True


# ------------------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------------------
# Standard scientific Python imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def classify(table, df, samples):
    X = []
    y = []
    all_data = {}
    y_index = table.y[-1].uid
    # c_cols = deepcopy(table.header)
    lime_xpoints = []

    for row in table.encodedrows:
        X_row = []
        y_row = -1
        for i, val in enumerate(row):
            if i == y_index:  # for multiple y if i in y_indexes:
                y_row = val
            else:
                X_row.append(val)
        X.append(X_row)
        y.append(y_row)
        # print(len(X[0]))

    for _ in range(1):
        perturbed_xtrain = np.random.normal(0,1,size=np.shape(X))
        p_train_x = np.vstack((X, X + perturbed_xtrain))
        # p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

        # p_train_x[:, c_cols] = X[:, c_cols]

        # for row in p_train_x:
        #     for c in c_cols:
        #         row[c] = np.random.choice(p_train_x[:,c])

        lime_xpoints.append(p_train_x)
        # all_y.append(p_train_y)
        # all_x = np.vstack(all_x)
        # all_y = np.concatenate(all_y)
    lime_points = np.vstack(lime_xpoints)
    #p = pertubated points & how far they move it...
    p = [1 for _ in range(len(lime_points))]
    #iid = independent variables with identical distributions & put a 0 for each var in the range of length of X
    iid = [0 for _ in range(len(X))]

    all_x = np.vstack((lime_points,X)).tolist()
    #creates an array of perturbed points & points with the same probability distribution
    #--> use for classifier; check to see if it's right
    all_y = np.array(p + iid).tolist()


    X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.20)
    #LR RF SVC
    clf = LogisticRegression(random_state=0)
    # clf = RandomForestClassifier(random_state=0)
    # clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)



    full = []
    for x in X_test:
        full.append(deepcopy(x))
    for j in range(len(y_test)):
        full[j].append(y_test[j])
        full[j].append(y_pred[j])
        full[j].append(samples)
        full[j].append(i)
    for row in full:
        a_series = pd.Series(row, index=df.columns)
        df = df.append(a_series, ignore_index=True)

    return df


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def clusterandclassify(csv, limiter=None):
    dataset = csv
    filename = dataset[:-4]  # cut off the .csv
    # colnames = ['accuracy', 'support', 'precision', 'recall', 'f1-score', 'precision_weighted', 'recall_weighted', 'f1-score_weighted', 'samples']
    # colnames = ['samples', 'accuracy', 'class0_precision', 'class0_recall',
                # 'class0_f1-score', 'class1_precision', 'class1_recall', 'class1_f1-score', 'macro_precision',
                # 'macro_recall', 'macro_f1-score']
    data = {}
    x_data = []

    lines = Table.readfile(r'./datasets/' + dataset)
    table = Table(1)
    table + lines[0]
    if limiter != None:
        for l in lines[1:limiter]:
            table + l
    else:
        for l in lines[1:]:
            table + l

    # print("Whole Data Classification...")
    table.encode_lines()
    # for col in table.protected:
    #     print("Protected:", str(col.name))
    #     if col in table.syms:
    #         print(str(col.name), ":", col.encoder.classes_)
    #
    # for col in table.y:
    #     if col in table.syms:
    #         print(str(col.name), ":", col.encoder.classes_)

    print(list(table.y[-1].encoder.classes_))

    columns = deepcopy(table.header)
    columns.append("predicted")
    columns.append("samples")
    columns.append("run_num")
    df2 = pd.DataFrame(columns=columns)
    df2 = classify(table, df2, len(table.rows))

    # print("Clustering ...")
    enough = int(math.sqrt(len(table.rows)))
    root = Table.clusters(table.rows, table, enough)

    # print("Sorting leaves ...")
    # print("Extrapolated Data Classification... until", (enough//2), "samples")
    # leafmedians(root) #bfs for the leaves gives median row
    treatments = [1,2,3,5, enough]
    # pbar = tqdm(list(range(1, (int(enough * 0.55)))))  # loading bar
    # list12 = [1]
    pbar = tqdm(treatments)  # loading bar
    for samples in pbar:
        pbar.set_description("Extrapolated Data Classification with %s samples" % samples)
        if samples == 1:
            MedianTable = leafmedians(root)
            # print("MT rows:", MedianTable.rows)
            df2 = classify(MedianTable, df2, samples)
        else:
            EDT = getLeafData(root, samples) #get x random point(s) from leaf clusters
            df2 = classify(EDT, df2, samples)

    # for key, v in data.items():
    #     # print("data dict: " , data)
    #     for key2 in v.items():
    #         # print("key2", key2)
    #         tmp = key2[1]
    #         x_data.append(tmp)

    # print("x_data ", x_data)
    # df = pd.DataFrame(x_data, columns=colnames)
    # print(df.head())
    final_columns = []
    for col in table.protected:
        final_columns.append(col.name)
    for col in table.klass:
        final_columns.append(col.name)
    # final_columns.append("GT")
    final_columns.append("predicted")
    final_columns.append("samples")
    final_columns.append("run_num")
    final_df = df2[final_columns]
    final_df.to_csv("./output/" + filename + "_lime_LR.csv", index=False)

    # df.to_csv("./output/" + filename + "_median_20runs_LR_testing.csv", index=False)

import cProfile

def main():
    random.seed(10019)
    datasets = ["diabetes.csv", "CleanCOMPAS53.csv", "GermanCredit.csv"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        clusterandclassify(dataset, limiter=1000)


    # clusterandclassify("diabetes.csv") #clusters
    # clusterandclassify("adultscensusincome.csv") #clusters
    # clusterandclassify("bankmarketing.csv") #clusters
    # clusterandclassify("CleanCOMPAS53.csv") #problem with empty cols?
    # clusterandclassify("GermanCredit.csv") #clusters
    # clusterandclassify("processed.clevelandhearthealth.csv") #clusters
    # clusterandclassify("defaultcredit.csv") #clusters
    # clusterandclassify("homecreditapplication_train.csv") # loaded 266113 rows after 2 hours; error on compiling sym/num cols


# self = options(__doc__)
if __name__ == '__main__':
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # pr.print_stats(sort='time')
