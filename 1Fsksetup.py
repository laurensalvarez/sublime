import copy,math
from collections import defaultdict
import sys
from tqdm import tqdm
import pandas as pd
from measure import measure_final_score,calculate_recall,calculate_precision,calculate_accuracy


def main():
    datasets = ["adultscensusincome.csv", "bankmarketing.csv", "defaultcredit.csv", "diabetes.csv", "CleanCOMPAS53.csv", "GermanCredit.csv"]
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        filename = dataset[:-4]
        df = pd.read_csv(r'./metrics/newDS/' + filename + "_sLR_metrics.csv")

        df1 = copy.deepcopy(df)

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

        feat = []
        if filename == "CleanCOMPAS53":
            feat.append("race(")
            feat.append("sex(")
        elif filename == "adultscensusincome":
            feat.append("race(")
            feat.append("sex(")
        elif filename == "GermanCredit":
            feat.append("sex(")
        elif filename == "defaultcredit":
            feat.append("SEX(")
        elif filename == "diabetes":
            feat.append("Age(")
        elif filename == "bankmarketing":
            feat.append("Age(")

        for s in sortedsamples:
            # print("Grouping DF-RF by", s, "samples: \n")
            dfRF2 = copy.deepcopy(df1)
            dfRF2.drop(dfRF2.loc[dfRF2['sample_size']!= s].index, inplace=True)

            features = copy.deepcopy(dfRF2["feature"].tolist())
            sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

            for f in sortedfeatures:
                if f in feat:
                    dfRF3 = copy.deepcopy(dfRF2)
                    dfRF3.drop(dfRF3.loc[dfRF3['feature']!= f].index, inplace=True)

                    recall = dfRF3 ['recall+']
                    precision = dfRF3 ['precision+']
                    accuracy = dfRF3 ['accuracy+']
                    F1_Score = dfRF3 ['F1_Score+']
                    AOD = dfRF3 ['AOD-']
                    EOD = dfRF3 ['EOD-']
                    SPD = dfRF3 ['SPD-']
                    FA0 = dfRF3 ['FA0-']
                    FA1 = dfRF3 ['FA1-']


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


        recall_df = pd.DataFrame(reformed_recalldict)
        recall_df.columns = ['_'.join(map(str, x)) for x in recall_df.columns]
        recall_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_recall+_.csv", header = None, index=True, sep=' ')

        prec_df = pd.DataFrame(reformed_predict)
        prec_df.columns = ['_'.join(map(str, x)) for x in prec_df.columns]
        prec_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_prec+_.csv", header = None, index=True, sep=' ')

        acc_df = pd.DataFrame(reformed_accdict)
        acc_df.columns = ['_'.join(map(str, x)) for x in acc_df.columns]
        acc_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_acc+_.csv", header = None, index=True, sep=' ')

        F1_df = pd.DataFrame(reformed_F1dict)
        F1_df.columns = ['_'.join(map(str, x)) for x in F1_df.columns]
        F1_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_F1+_.csv", header = None, index=True, sep=' ')

        AOD_df = pd.DataFrame(reformed_AODdict)
        AOD_df.columns = ['_'.join(map(str, x)) for x in AOD_df.columns]
        AOD_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_AOD-_.csv", header = None, index=True, sep=' ')

        EOD_df = pd.DataFrame(reformed_EODdict)
        EOD_df.columns = ['_'.join(map(str, x)) for x in EOD_df.columns]
        EOD_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_EOD-_.csv", header = None, index=True, sep=' ')

        SPD_df = pd.DataFrame(reformed_SPDdict)
        SPD_df.columns = ['_'.join(map(str, x)) for x in SPD_df.columns]
        SPD_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_SPD-_.csv", header = None, index=True, sep=' ')

        FA0_df = pd.DataFrame(reformed_FA0dict)
        FA0_df.columns = ['_'.join(map(str, x)) for x in FA0_df.columns]
        FA0_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_FA0-_.csv", header = None, index=True, sep=' ')

        FA1_df = pd.DataFrame(reformed_FA1dict)
        FA1_df.columns = ['_'.join(map(str, x)) for x in FA1_df.columns]
        FA1_df.transpose().to_csv("./sk_data/newDS/" + filename + "_sLR_FA1-_.csv", header = None, index=True, sep=' ')


if __name__ == '__main__':
    main()
