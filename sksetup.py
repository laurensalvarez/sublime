import copy,math
from collections import defaultdict
import sys
from tqdm import tqdm
import pandas as pd
from measure import measure_final_score,calculate_recall,calculate_precision,calculate_accuracy


def main():
    datasets = ["diabetes.csv", "CleanCOMPAS53.csv", "GermanCredit.csv"]
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        filename = dataset[:-4]
        df = pd.read_csv(r'./metrics/copies/' + filename + "_LR_cmetrics.csv")

        df1 = copy.deepcopy(df)
        # df15 = df1.head(3)
        # print(df15)
        # sys.exit()

        samples = copy.deepcopy(df1["sample_size"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))

        recalldict = defaultdict(dict)
        precisiondict = defaultdict(dict)
        accdict = defaultdict(dict)
        F1dict = defaultdict(dict)
        AODdict = defaultdict(dict)
        EODdict = defaultdict(dict)
        SPDdict = defaultdict(dict)
        FA0dict = defaultdict(dict)
        FA1dict = defaultdict(dict)

        for s in sortedsamples:
            # print("Grouping DF-RF by", s, "samples: \n")
            dfRF2 = copy.deepcopy(df1)
            dfRF2.drop(dfRF2.loc[dfRF2['sample_size']!= s].index, inplace=True)

            features = copy.deepcopy(dfRF2["feature"].tolist())
            sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

            for f in sortedfeatures:
                # print("Grouping DF-RF by feature", f, " \n")
                dfRF3 = copy.deepcopy(dfRF2)
                dfRF3.drop(dfRF3.loc[dfRF3['feature']!= f].index, inplace=True)

                recall = dfRF3 ['recall']
                precision = dfRF3 ['precision']
                accuracy = dfRF3 ['accuracy']
                F1_Score = dfRF3 ['F1_Score']
                AOD = dfRF3 ['AOD']
                EOD = dfRF3 ['EOD']
                SPD = dfRF3 ['SPD']
                FA0 = dfRF3 ['FA0']
                FA1 = dfRF3 ['FA1']


                recalldict[s][f] = recall.values
                precisiondict[s][f] = precision.values
                accdict[s][f] = accuracy.values
                F1dict[s][f] = F1_Score.values
                AODdict[s][f] = AOD.values
                EODdict[s][f] = EOD.values
                SPDdict[s][f] = SPD.values
                FA0dict[s][f] = FA0.values
                FA1dict[s][f] = FA1.values


        reformed_recalldict = {}
        for outerKey, innerDict in recalldict.items():
            for innerKey, values in innerDict.items():
                reformed_recalldict[(outerKey,innerKey)] = values

        reformed_predict = {}
        for outerKey, innerDict in precisiondict.items():
            for innerKey, values in innerDict.items():
                reformed_predict[(outerKey,innerKey)] = values

        reformed_accdict = {}
        for outerKey, innerDict in accdict.items():
            for innerKey, values in innerDict.items():
                reformed_accdict[(outerKey,innerKey)] = values

        reformed_F1dict = {}
        for outerKey, innerDict in F1dict.items():
            for innerKey, values in innerDict.items():
                reformed_F1dict[(outerKey,innerKey)] = values

        reformed_AODdict = {}
        for outerKey, innerDict in AODdict.items():
            for innerKey, values in innerDict.items():
                reformed_AODdict[(outerKey,innerKey)] = values

        reformed_EODdict = {}
        for outerKey, innerDict in EODdict.items():
            for innerKey, values in innerDict.items():
                reformed_EODdict[(outerKey,innerKey)] = values

        reformed_SPDdict = {}
        for outerKey, innerDict in SPDdict.items():
            for innerKey, values in innerDict.items():
                reformed_SPDdict[(outerKey,innerKey)] = values

        reformed_FA0dict = {}
        for outerKey, innerDict in FA0dict.items():
            for innerKey, values in innerDict.items():
                reformed_FA0dict[(outerKey,innerKey)] = values

        reformed_FA1dict = {}
        for outerKey, innerDict in FA1dict.items():
            for innerKey, values in innerDict.items():
                reformed_FA1dict[(outerKey,innerKey)] = values


        recall_df = pd.DataFrame(reformed_recalldict).transpose()
        recall_df.to_csv("./sk_data/" + filename + "_LR_recall_all.csv", index=True)

        prec_df = pd.DataFrame(reformed_predict).transpose()
        prec_df.to_csv("./sk_data/" + filename + "_LR_prec_all.csv", index=True)

        acc_df = pd.DataFrame(reformed_accdict).transpose()
        acc_df.to_csv("./sk_data/" + filename + "_LR_acc_all.csv", index=True)

        F1_df = pd.DataFrame(reformed_F1dict).transpose()
        F1_df.to_csv("./sk_data/" + filename + "_LR_F1_all.csv", index=True)

        AOD_df = pd.DataFrame(reformed_AODdict).transpose()
        AOD_df.to_csv("./sk_data/" + filename + "_LR_AOD_all.csv", index=True)

        EOD_df = pd.DataFrame(reformed_EODdict).transpose()
        EOD_df.to_csv("./sk_data/" + filename + "_LR_EOD_all.csv", index=True)

        SPD_df = pd.DataFrame(reformed_SPDdict).transpose()
        SPD_df.to_csv("./sk_data/" + filename + "_LR_SPD_all.csv", index=True)

        FA0_df = pd.DataFrame(reformed_FA0dict).transpose()
        FA0_df.to_csv("./sk_data/" + filename + "_LR_FA0_all.csv", index=True)

        FA1_df = pd.DataFrame(reformed_FA1dict).transpose()
        FA1_df.to_csv("./sk_data/" + filename + "_LR_FA1_all.csv", index=True)


if __name__ == '__main__':
    main()
