

class Abcd:
  def __init__(i,db="all",rx="all"):
    i.db = db; i.rx=rx;
    i.yes = i.no = 0
    i.known = {}; i.a= {}; i.b= {}; i.c= {}; i.d= {}

  def __call__(i,actual=None,predicted=None):
    return i.keep(actual,predicted)

  def tell(i,actual,predict):
    i.knowns(actual)
    i.knowns(predict)
    if actual == predict: i.yes += 1
    else                :  i.no += 1
    for x in  i.known:
      if actual == x:
        if  predict == actual: i.d[x] += 1
        else                 : i.b[x] += 1
      else:
        if  predict == x     : i.c[x] += 1
        else                 : i.a[x] += 1

  def knowns(i,x):
    if not x in i.known:
      i.known[x]= i.a[x]= i.b[x]= i.c[x]= i.d[x]= 0.0
    i.known[x] += 1
    if (i.known[x] == 1):
      i.a[x] = i.yes + i.no

  def header(i):
    print("#",('{0:20s} {1:11s}  {2:7s}  {3:7s} {4:7s} '+ \
           '{5:7s}{6:7s} {7:8s} {8:8s} {9:8s} '+ \
           '{10:8s} {11:8s}{12:8s}{13:10s}').format(
      "db", "rx",
     "n", "a","b","c","d","acc","pd","pf","prec",
      "f","g","class"))
    print('-'*150)

  def ask(i):
    def p(y) : return int(100*y + 0.5)
    def n(y) : return int(y)
    pd = pf = pn = prec = g = f = acc = 0
    for x in i.known:
      a= i.a[x]; b= i.b[x]; c= i.c[x]; d= i.d[x]
      if (b+d)    : pd   = d     / (b+d)
      if (a+c)    : pf   = c     / (a+c)
      if (a+c)    : pn   = (b+d) / (a+c)
      if (c+d)    : prec = d     / (c+d)
      if (1-pf+pd): g    = 2*(1-pf)*pd / (1-pf+pd)
      if (prec+pd): f    = 2*prec*pd/(prec+pd)
      if (i.yes + i.no): acc= i.yes/(i.yes+i.no)
      print("#",('{0:20s} {1:10s} {2:7d} {3:7d} {4:7d} '+ \
          '{5:7d} {6:7d} {7:7d} {8:7d} {9:7d} '+ \
         '{10:7d} {11:7d} {12:7d} {13:10d}').format(i.db,
          i.rx,  n(b + d), n(a), n(b),n(c), n(d),
          p(acc), p(pd), p(pf), p(prec), p(f), p(g),x))
      #print x,p(pd),p(prec)

"""
output:
 $ python abcd.py
# db                   rx           n     a    b    c   d    acc pd  pf  prec f  g  class
----------------------------------------------------------------------------------------------------
# randomIn             jiggle       22    0    5    5   17   63  77 100  77  77   0 a
# randomIn             jiggle        5   17    5    5    0   63   0  23   0  77   0 b
"""



if __name__ == "__main__":
  import random
  from tqdm import tqdm
  import pandas as pd

  datasets = ["adultscensusincome.csv", "bankmarketing.csv", "defaultcredit.csv", "diabetes.csv", "CleanCOMPAS53.csv", "GermanCredit.csv"]
  pbar = tqdm(datasets)
  for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        filename = dataset[:-4]
        filepath = r'./output/mid2/' + filename + "_mid2RF.csv"
        abcd = Abcd(db=filename,rx='jiggle')

        df = pd.read_csv(filepath)
        if '!probability' in df.columns:
            train = df["!probability"].tolist()
        elif '!Probability' in df.columns:
            train = df["!Probability"].tolist()

        test = df["predicted"].tolist()

        print ("test", len(test))
        print("train", len(train))

        for actual, predicted in zip(train,test):
            abcd.tell(actual,predicted)
        abcd.header()
        abcd.ask()
        print("\n\n")
