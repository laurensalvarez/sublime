import copy
import math
import sys, random
from tqdm import tqdm
import pandas as pd
import numpy as np





def getMedians(path, metric, mediandf):
    skdf = pd.read_csv(path, sep = ' ', header=None, index_col=0, skip_blank_lines=True)
    skdf["median"] = skdf.median(axis = 1)
    skdf["features"] = skdf.index
    medianValues = skdf["median"].values
    fValues = skdf["features"].values

    # mediandf[metric] = medianValues
    mediandf[metric] = pd.Series(medianValues, index=skdf.index)
    mediandf["features"] = pd.Series(fValues, index=skdf.index)
    mediandf["feature"] = pd.Series(fValues, index=skdf.index)
    # mediandf["features"] = skdf["features"].values
    # mediandf["feature"] = skdf["features"].values
    mediandf.set_index("feature", inplace = False)


    # df[metric] = metricdf["median"].values
    # df["features"] = metricdf["features"].values
    # df["feature"] = metricdf["features"].values
    # df.set_index("features", inplace = True)
    # print(df)
    # print ("in loop" , mediandf)

    return mediandf





if __name__ == "__main__":
    random.seed(1)
    datasets = ["adultscensusincome.csv", "bankmarketing.csv", "defaultcredit.csv", "diabetes.csv", "CleanCOMPAS53.csv", "GermanCredit.csv"]
    metrics = ['recall+', 'prec+', 'acc+', 'F1+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-']
    pbar = tqdm(datasets)

    for n in [1,2,3,5]:
        n = str(n)
        columns = metrics.copy()
        columns.insert(0, "features")
        columns.insert(0, "dataset")
        fulldf = pd.DataFrame(columns=columns)
        datasetdf = pd.DataFrame(columns=columns)
        datasetsdf = pd.DataFrame(columns=columns)

        for dataset in pbar:
            pbar.set_description("Processing %s" % dataset)
            filename = dataset[:-4]
            mediandf = pd.DataFrame(columns=columns)

            for m in metrics:
                print("\n" +"-" + filename +"-" + m + "\n"  )
                path =  "./sk_data/EM_MODE_C0/" + filename +"_" + n + "_LR_" + m +"_.csv"
                mediandf = getMedians(path, m, mediandf)

            multi = len(mediandf.index)
            namelist = [filename] * multi
            mediandf["dataset"] = namelist
            # print(mediandf)

            mediandf.to_csv("./medians/EM_MODE_C0/" + n + "/" + filename + "_" + n + "_LR_medians.csv", index = False)
            datasetdf = pd.concat([datasetdf, mediandf], ignore_index=True)
        # print(datasetdf)

        fulldf = datasetdf[columns]
        fulldf.to_csv("./medians/EM_MODE_C0/"  + n + "_EM_MODE_LR_medians.csv", index = False)
