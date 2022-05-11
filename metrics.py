import numpy as np
import copy,math
import math
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from measure import measure_final_score,calculate_recall,calculate_precision,calculate_accuracy
from columns import Table, Col, Sym, Num



###############################################
### confusion matrix
###############################################

# normal matrix
def getMetrics(test_df, y_true, y_pred, biased_col, samples):

    TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()

    recall = measure_final_score(test_df, y_true, y_pred, biased_col, 'recall')
    precision = measure_final_score(test_df, y_true, y_pred, biased_col, 'precision')
    accuracy = measure_final_score(test_df, y_true, y_pred, biased_col, 'accuracy')
    F1 = measure_final_score(test_df, y_true, y_pred, biased_col, 'F1')
    AOD = measure_final_score(test_df, y_true, y_pred, biased_col, 'aod')
    EOD =measure_final_score(test_df, y_true, y_pred, biased_col, 'eod')
    SPD = measure_final_score(test_df, y_true, y_pred, biased_col, 'SPD')

    # print("recall :", recall)
    # print("precision :", precision)
    # print("accuracy :", accuracy)
    # print("F1 Score :", F1)
    # print("AOD :" AOD)
    # print("EOD :" + biased_col , EOD)
    # print("SPD:", SPD)

    return [recall, precision, accuracy, F1, AOD, EOD, SPD, biased_col, samples]


def makeBinary(preddf, dataset):
    if dataset == "diabetes":
        preddf['Age('] = np.where((preddf['Age('] > 25), 0, 1)
        # preddf.replace({'Age(':{'Malaysia' : 56, 'Paris' : 778 }})
        # preddf['Age('] = np.where((preddf['Age('] <= 25), 1, preddf['Age('])

    if dataset == "CleanCOMPAS53":
        # preddf['race('] = np.where(preddf['race('] == 0 or 1 or 3 or 4 or 5, 0, preddf['race('])
        preddf['race('] = np.where((preddf['race('] == 2), 1, 0)

        preddf['Age('] = np.where((preddf['Age('] > 25), 0, 1)

    if dataset == "GermanCredit":
        preddf['savings('] = np.where((preddf['savings('] == 0) | (preddf['savings('] == 1) | (preddf['savings('] ==  4), 0, 1)
        # preddf['savings('] = np.where(preddf['savings('] == 2 or 3, 1, preddf['savings('])

        preddf['Age('] = np.where((preddf['Age('] > 25), 0, 1)
        # preddf['Age('] = np.where(preddf['Age('] > 25, 0, preddf['Age('])
        # preddf['Age('] = np.where(preddf['Age('] <= 25, 1, preddf['Age('])

        preddf['sex('] = np.where((preddf['sex('] == 1), 0, 1)
        # preddf['sex('] = np.where(preddf['sex('] == 0 or 2 or 3, 1, preddf['sex('])

    return preddf

def getColNames(table):
    colNames = []
    for col in table.cols:
        colNames.append(col.name)
    return colNames

def getBiasCols(dataset):
    bias_cols = []
    if dataset == "CleanCOMPAS53":
        bias_cols = ["sex(", "Age(","race("]

    if dataset == "GermanCredit":
        bias_cols = ["savings(", "sex(" , "Age(", "foreign_worker("]

    if dataset == "diabetes":
        bias_cols = ["Age("]

    return bias_cols


def sampleMetrics(test_df, y_true, y_pred, biased_cols, samples):
    colmetrics = {}

    for i in biased_cols:
        colmetrics[i] = getMetrics(test_df, y_true, y_pred, i, samples)

    return colmetrics

###############################################
###
###############################################
def main():
    datasets = ["diabetes.csv", "CleanCOMPAS53.csv", "GermanCredit.csv"]
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        filename = dataset[:-4]
        predlines = Table.readfile(r'./output/' + filename + "_protected_predictions2.csv")

        predtable = Table(2)
        predtable + predlines[0]
        for l in predlines[1:]:
            predtable + l

        predColNames = getColNames(predtable)

        preddf = pd.DataFrame(predtable.rows, columns=predColNames)

        bintdf = makeBinary(preddf, filename)
        # print(bintdf)
        # print(bintdf["Age("])
        #
        # sys.exit()

        biased_cols = getBiasCols(filename)

        # using sorted() + set() + count()
        # sorting and removal of duplicates
        samples = copy.deepcopy(bintdf["samples"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))

        all_metrics = {}
        rows = []
        for s in sortedsamples:
            print("Metrics for", s, "samples: \n")
            dfs = copy.deepcopy(bintdf)
            dfs.drop(dfs.loc[dfs['samples']!= s].index, inplace=True)
            y_true = dfs["!Probability"]
            y_pred = dfs["predicted"]
            all_metrics[s] = sampleMetrics(dfs, y_true, y_pred, biased_cols, s)

        for key, v in all_metrics.items():
            # print("data dict: " , data)
            for key2 in v.items():
                # print("key2", key2)
                tmp = key2[1]
                print(tmp)
                rows.append(tmp)

        fulldf = pd.DataFrame(rows, columns = ['recall', 'precision', 'accuracy', 'F1 Score', 'AOD', 'EOD', 'SPD','feature', 'sample size'])

        fulldf.to_csv("./metrics/" + filename + "_SVM_metrics.csv", index=False)



# self = options(__doc__)
if __name__ == '__main__':
    main()
