import math, re, random, statistics, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cProfile
from collections import defaultdict
from copy import deepcopy
from itertools import count
from tqdm import tqdm

from columns import Table, Col, Sym, Num, TreeNode

from adversarial_models import *
from utils import *

import lime
import lime.lime_tabular

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

def sortedleafclusterlabels(root, f,
                            how=None):  # for all of the leaves from smallest to largest print len of rows & median
    clabel = None
    xlabel = None
    match = 0
    counter = 0

    for leaf in sorted(nodes(root), key=how or rowSize):
        counter += 1
        t = leaf.leftTable

        clabel = t.y[0].mid()

        for row in t.rows:
            if row not in t.skip:
                xlabel = str(row[len(row) - 1])
                if xlabel == clabel:  # this will crash if the xlabel is a string and the clabel is an int (i.e GermanCredit)
                    match += 1

        t.clabels = [clabel for i in range(len(t.rows))]
        matches = match / (len(t.rows))

        f.write("Leaf " + str(counter) + "\n")
        if matches >= 0.80:
            f.write("--------------------------------> Good Cluster Label <--------" + "\n")
        else:
            f.write("Bad Cluster Label" + "\n")

        percent = "{0:.0%}".format(matches, 2)
        f.write("Cluster Label: " + str(clabel) + "\n")
        f.write("Label Matches: " + str(match) + "/" + str(len(t.rows)) + "\n")
        f.write("Label Matches Percentage: " + str(percent) + "\n")
        f.write(
            "---------------------------------------------------------------------------------------------------------------------------------------" + "\n")
        f.write(
            "---------------------------------------------------------------------------------------------------------------------------------------" + "\n")

        match = 0

    def dump(self, f):
        # DFS
        if self.leaf:
            f.write("Dump Leaf Node: " + str(self.uid) + "\n")
            f.write("Dump Leaf Table: " + "\n")
            self.leftTable.dump(f)
            return

        if self.leftNode is not None:
            self.leftNode.dump(f)
        if self.rightNode is not None:
            self.rightNode.dump(f)

    def csvDump(self, f):
        # DFS
        if self.leaf:
            self.leftTable.csvDump(f)
            return

        if self.leftNode is not None:
            self.leftNode.csvDump(f)

        if self.rightNode is not None:
            self.rightNode.csvDump(f)


def isValid(self, row):
    for val in row:
        if val == '?':
            return False
    return True


def getXY(table):
    X = []
    y = []
    y_index = table.y[-1].uid

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
    return X,y


def classify(table, df, X_test, y_test, samples, f):
    i = 1
    full = []

    X_train, y_train = getXY(table)

    #LR RF SVC
    clf = LogisticRegression(random_state=0)
    # clf = RandomForestClassifier(random_state=0)
    # clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_pred_list = y_pred.tolist()
    y_test_list = y_test.tolist()
    for x in list(X_test.values):
        full.append(deepcopy(x))
    for j in range(len(y_test_list)):
        full[j] = np.append(full[j],y_test_list[j])
        full[j] = np.append(full[j],y_pred_list[j])
        full[j] = np.append(full[j], samples)
        full[j] = np.append(full[j], f)
        full[j] = np.append(full[j], i)
    for row in full:
        a_series = pd.Series(row, index=df.columns)
        df = df.append(a_series, ignore_index=True)
    return df


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def getTable(csv, limiter = None):
    dataset = csv
    filename = dataset[:-4]  # cut off the .csv
    lines = Table.readfile(r'./datasets/' + dataset)
    table = Table(1)
    table + lines[0]
    if limiter != None:
        for l in lines[1:limiter]:
            table + l
    else:
        for l in lines[1:]:
            table + l

    return table, filename

def clusterandclassify(table, filename):
    table.encode_lines()
    y_index = table.y[-1].name
    tcolumns = deepcopy(table.header)
    trows = deepcopy(table.encodedrows)
    # dsdf = pd.DataFrame(data= trows, columns=tcolumns)

    tcols = deepcopy(table.header)
    tcols.append("predicted")
    tcols.append("samples")
    tcols.append("fold")
    tcols.append("run_num")
    sampledf = pd.DataFrame(columns=tcols)
    full_df = pd.DataFrame(columns=tcols)

    # pertub the points before the kfold split
    # pertub the full table and then split? bc we're trying to classify for
    # pertubed vs OG not the classes anymore...
    lime_xpoints = []
        for _ in range(1):
            perturbed_xtrain = np.random.normal(0,1,size=np.shape(trows))
            p_train_x = np.vstack((trows, trows + perturbed_xtrain))
            lime_xpoints.append(p_train_x)

        lime_points = np.vstack(lime_xpoints)
        #p = pertubated points & how far they move it...
    p = [1 for _ in range(len(lime_points))]
    #iid = independent variables with identical distributions & put a 0 for each var in the range of length of X
    iid = [0 for _ in range(len(trows))]

    all_x = np.vstack((lime_points,trows)).tolist()
    #creates an array of perturbed points & points with the same probability distribution
    #--> use for classifier; check to see if it's right
    all_y = np.array(p + iid).tolist()

    dsdf = pd.DataFrame(data= all_x, columns=tcolumns)
    dsdf.drop(columns = [y_index])
    dsdf['yvals'] = all_y
    print (dsdf.head)

    sys.exit()

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=222)
    # drop the true y vals bc we're not interested in the orginal class anymore

    f = 1 #do we want to count the folds??
    for train_index, test_index in rkf.split(dsdf):
        X_train = dsdf.iloc[train_index].drop(columns = [y_index])
        X_test = dsdf.iloc[test_index].drop(columns = [y_index])
        y_train = dsdf.iloc[train_index][y_index]
        y_test = dsdf.iloc[test_index][y_index]

        table2 = Table(10)
        nprows = X_train.values
        header = list(X_train.columns.values)
        table2 + header
        for l in nprows:
            table2 + l

        enough = int(math.sqrt(len(table2.rows)))
        root = Table.clusters(table2.rows, table, enough)

        treatments = [1,2,3,5,enough]
        for samples in treatments:
            if samples == 1:
                MedianTable = leafmedians(root)
                sampledf = classify(MedianTable, sampledf, X_test, y_test, samples, f)
            else:
                EDT = getLeafData(root, samples) #get x random point(s) from leaf clusters
                sampledf = classify(EDT, sampledf, X_test, y_test, samples, f)
            full_df = full_df.append(sampledf)
        f += 1
        print("f:", f)
    # print("full_df head:", full_df.head)
        # final_df2 = pd.concat([final_df2, final_df], ignore_index=False)
    final_columns = []
    for col in table.protected:
        final_columns.append(col.name)
    for col in table.klass:
        final_columns.append(col.name)
    final_columns.append("predicted")
    final_columns.append("samples")
    final_columns.append("fold")
    final_columns.append("run_num")
    output_df = full_df[final_columns]
    output_df.to_csv("./output/fold/" + filename + "_folded_SVM.csv", index=False)


def main():
    random.seed(10019)
    datasets = ["diabetes.csv"] #, "GermanCredit.csv", "CleanCOMPAS53.csv"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        table, filename = getTable(dataset, limiter=1000)
        clusterandclassify(table, filename)


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
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    pr.print_stats(sort='time')
