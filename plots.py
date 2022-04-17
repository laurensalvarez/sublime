import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(dataset,colnames):
    df = pd.DataFrame(dataset, columns=colnames)
    # print(df.head())

    df.to_csv("./output/"+filename + "_SVM.csv", index=False)

    plt.figure(1)
    # macro_ax = df.plot(x = ['samples','samples','samples','samples'], y = ['accuracy','macro_precision','macro_recall', 'macro_f1-score'] , color = ['blue','orange','green', 'red'],kind = 'scatter', marker = ',')
    macro_ax = df.plot(x = 'samples', y = 'accuracy', kind = 'scatter', c = 'b', marker = ',')
    df.plot(x = 'samples', y = 'macro_precision', kind = 'scatter', c = 'orange', marker = ',', ax = macro_ax)
    df.plot(x = 'samples', y = 'macro_recall', kind = 'scatter', c = 'g', marker = ',', ax = macro_ax)
    df.plot(x = 'samples', y = 'macro_f1-score', kind = 'scatter', c = 'r', marker = ',', ax = macro_ax)
    dfmeans = df.groupby(['samples'])['accuracy','macro_precision','macro_recall', 'macro_f1-score'].mean().plot(marker = 'o',ls = '-', ax = macro_ax)
    macro_ax.set_xlabel("sample size")
    macro_ax.set_ylabel("macro precision, recall, f1-score, and accuracy")
    plt.savefig("./plots/"+filename + "macro_SVM.png")
    plt.close()

    plot2 = plt.figure(2)
    class0_ax = df.plot(x = 'samples', y = 'accuracy', kind = 'scatter', c = 'b', marker = ',')
    df.plot(x = 'samples', y = 'class0_precision', kind = 'scatter', c = 'orange', marker = ',', ax = class0_ax)
    df.plot(x = 'samples', y = 'class0_recall', kind = 'scatter', c = 'g', marker = ',', ax = class0_ax)
    df.plot(x = 'samples', y = 'class0_f1-score', kind = 'scatter', c = 'r', marker = ',', ax = class0_ax)
    negdfmeans = df.groupby(['samples'])['accuracy','class0_precision', 'class0_recall', 'class0_f1-score'].mean().plot(marker = 'o', ls = '-', ax = class0_ax)
    class0_ax.set_xlabel("sample size")
    class0_ax.set_ylabel(" class 0: precision, recall, f1-score, and accuracy")
    plt.savefig("./plots/"+filename + "class_0_SVM.png")
    plt.close()

    plot3 = plt.figure(3)
    class1_ax = df.plot(x = 'samples', y = 'accuracy', kind = 'scatter', c = 'b', marker = ',')
    df.plot(x = 'samples', y = 'class1_precision', kind = 'scatter', c = 'orange', marker = ',', ax = class1_ax)
    df.plot(x = 'samples', y = 'class1_recall', kind = 'scatter', c = 'g', marker = ',', ax = class1_ax)
    df.plot(x = 'samples', y = 'class1_f1-score', kind = 'scatter', c = 'r', marker = ',', ax = class1_ax)
    posdfmeans = df.groupby(['samples'])['accuracy','class1_precision', 'class1_recall', 'class1_f1-score'].mean().plot(marker = 'o', ls = '-', ax = class1_ax)
    class1_ax.set_xlabel("sample size")
    class1_ax.set_ylabel(" class 1: precision, recall, f1-score, and accuracy")
    plt.savefig("./plots/"+filename + "class_1_SVM.png")
    plt.close()
