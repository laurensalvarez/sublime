from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import columns

def plot(data, colnames, filename):
    print("data", data)
    df = pd.read_csv(r'./output/' + data)
    # print(df.head())

    plt.figure(1)
    # macro_ax = df.plot(x = ['samples','samples','samples','samples'], y = ['accuracy','macro_precision','macro_recall', 'macro_f1-score'] , color = ['blue','orange','green', 'red'],kind = 'scatter', marker = ',')
    macro_ax = df.plot(x = 'samples', y = 'accuracy', kind = 'scatter', c = 'b', marker = ',', alpha = 1/5)
    df.plot(x = 'samples', y = 'macro_precision', kind = 'scatter', c = 'orange', marker = ',', ax = macro_ax, alpha = 1/5)
    df.plot(x = 'samples', y = 'macro_recall', kind = 'scatter', c = 'g', marker = ',', ax = macro_ax, alpha = 1/5)
    df.plot(x = 'samples', y = 'macro_f1-score', kind = 'scatter', c = 'r', marker = ',', ax = macro_ax, alpha = 1/5)
    dfmeans = df.groupby(['samples'])['accuracy','macro_precision','macro_recall', 'macro_f1-score'].mean().plot(marker = 'o',ls = '-', ax = macro_ax, alpha = 1/5)
    macro_ax.set_xlabel("sample size")
    macro_ax.set_ylabel("macro precision, recall, f1-score, and accuracy")
    plt.savefig("./plots/"+filename + "macro_SVM.png")
    plt.close()

    plot2 = plt.figure(2)
    class0_ax = df.plot(x = 'samples', y = 'accuracy', kind = 'scatter', c = 'b', marker = ',', alpha = 1/5)
    df.plot(x = 'samples', y = 'class0_precision', kind = 'scatter', c = 'orange', marker = ',', ax = class0_ax,alpha = 1/5)
    df.plot(x = 'samples', y = 'class0_recall', kind = 'scatter', c = 'g', marker = ',', ax = class0_ax, alpha = 1/5)
    df.plot(x = 'samples', y = 'class0_f1-score', kind = 'scatter', c = 'r', marker = ',', ax = class0_ax, alpha = 1/5)
    negdfmeans = df.groupby(['samples'])['accuracy','class0_precision', 'class0_recall', 'class0_f1-score'].mean().plot(marker = 'o', ls = '-', ax = class0_ax, alpha = 1/5)
    class0_ax.set_xlabel("sample size")
    class0_ax.set_ylabel(" class 0: precision, recall, f1-score, and accuracy")
    plt.savefig("./plots/"+filename + "class_0_SVM.png")
    plt.close()

    plot3 = plt.figure(3)
    class1_ax = df.plot(x = 'samples', y = 'accuracy', kind = 'scatter', c = 'b', marker = ',', alpha = 1/5)
    df.plot(x = 'samples', y = 'class1_precision', kind = 'scatter', c = 'orange', marker = ',', ax = class1_ax, alpha = 1/5)
    df.plot(x = 'samples', y = 'class1_recall', kind = 'scatter', c = 'g', marker = ',', ax = class1_ax, alpha = 1/5)
    df.plot(x = 'samples', y = 'class1_f1-score', kind = 'scatter', c = 'r', marker = ',', ax = class1_ax, alpha = 1/5)
    posdfmeans = df.groupby(['samples'])['accuracy','class1_precision', 'class1_recall', 'class1_f1-score'].mean().plot(marker = 'o', ls = '-', ax = class1_ax, alpha = 1/5)
    class1_ax.set_xlabel("sample size")
    class1_ax.set_ylabel(" class 1: precision, recall, f1-score, and accuracy")
    plt.savefig("./plots/"+filename + "class_1_SVM.png")
    plt.close()


def main():
    random.seed(10019)
    datasets = ["diabetes_median_SVM.csv", "CleanCOMPAS53_median_SVM.csv", "GermanCredit_median_SVM.csv"]
    colnames = ['samples','accuracy', 'class0_precision', 'class0_recall',
    'class0_f1-score','class1_precision', 'class1_recall', 'class1_f1-score', 'macro_precision', 'macro_recall', 'macro_f1-score']
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        # lines = columns.Table.readfile(r'./output/' + dataset)
        # table = columns.Table(1)
        # table + lines[0]
        filename = dataset[:-4]
        plot(dataset, colnames, filename)


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
    main()
