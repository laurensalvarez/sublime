from collections import defaultdict
from sklearn import preprocessing
import math
import re
import random
import statistics
import sys
from itertools import count



# ------------------------------------------------------------------------------
# Column Class
# ------------------------------------------------------------------------------
class Col:
    def __init__(self, name): #The __init__ method lets the class initialize the object's attributes and serves no other purpose
        self._id = 0 # underscore means hidden var
        self.name = name # self is an instance of the class with all of the attributes & methods

    def __add__(self,v):
        return v

    def diversity(self):
        return 0

    def mid(self):
        return 0

    def dist(self, x, y):
        return 0

    @staticmethod
    def id(self):
        self._id += 1
        return self._id

# ------------------------------------------------------------------------------
# Symbolic Column Class
# ------------------------------------------------------------------------------
class Sym(Col):
    def __init__(self,name,uid,data=None): #will override Col inheritance (could use super() to inherit)
        Col.__init__(self,name) #invoking the __init__ of the parent class; If you forget to invoke the __init__() of the parent class then its instance variables would not be available to the child class.
        self.n = 0
        self.most = 0
        self.mode = ""
        self.uid = uid #uid --> it allows for permanence and recalling necessary subtables
        self.count = defaultdict(int) #will never throw a key error bc it will guve default value as missing key
        self.encoder = preprocessing.LabelEncoder()
        self.coltype = 0
        self.vals = []
        if data != None: #initializes the empty col with val
            for val in data:
                self + val #calls __add__

    #def __str__(self):
        #print to a sym; overrides print(); could replace dump() TBD

    def __add__(self, v): return self.add(v,1) #need to adds? i forgot why

    def add (self, v, inc=1): #want to be able to control the increments
        self.n += inc # add value to the count
        self.vals.append(v)
        self.count[v] += inc # add value to the dictionary with count +1
        tmp = self.count[v]
        if tmp > self.most: #check which is the most seen; if it's the most then assign and update mode
            self.most, self.mode = tmp, v # a,b = b,a
        return v

    def diversity(self): #entropy of all of n
        # is a measure of the randomness in the information being processed.
        # The higher the entropy, the harder it is to draw any conclusions from
        # that information.
        e = 0
        for k, v in self.count.items(): #iterate through a list of tuples (key, value)
            p = v/self.n
            e -= p*math.log(p)/math.log(2)
        return e

    def mid(self): #midpoint as mode (which sym appears the most)
        return self.mode

    def dist(self, x, y): #Aha's distance between two syms
        if (x == "?" or x == "") or (y == "?" or y == ""): #check if the empty is just a bug
            return 1
        return 0 if x == y else 1

# ------------------------------------------------------------------------------
# Numeric Column Class
# ------------------------------------------------------------------------------
big = sys.maxsize
tiny = 1/big

class Num(Col):
    def __init__(self, name, uid, data=None):
        Col.__init__(self, name)
        self.n = 0
        self.mu = 0 #
        self.m2 = 0 # for moving std dev
        self.sd = 0
        self.lo = big #float('inf')
        self.hi = -big #-float('inf')
        self.vals = []
        self.uid = uid
        self.median = 0
        self.coltype = 1
        if data != None:
            for val in data:
                self + val #calls __add__

    def __add__(self, v):
        #add the column; calculate the new lo/hi, get the sd using the 'chasing the mean'
        self.n += 1
        self.vals.append(v) #add value to a list
        try:
            if v < self.lo: #if the val is < the lowest; reassign
                self.lo = v
            if v > self.hi: #if the val is > the highest; reassign
                self.hi = v
            d = v - self.mu #distance to the mean
            self.mu += d / self.n # normalize it
            self.m2 += d * (v - self.mu) # calculate momentumn of the series in realtionship to the distance
            self.sd = self._numSd() #
            self.median = self.mid()
        except:
            print("failed col name:", self.name, "id:" , self.uid)
        return v

    def _numSd(self):
        # Standard deviation is a number that describes how
        # spread out the values are. A low standard deviation
        # means that most of the numbers are close to the mean (average) value.
        # A high standard deviation means that the values are spread out over a wider range.
        if self.m2 < 0:  #means there's is no momentum to the series
            return 0
        if self.n < 2: #if there's two items
            return 0
        return math.sqrt(self.m2/(self.n -1)) #calculate std dev

    def diversity(self): #return standard dev
        return self.sd

    def mid(self): #get midpoint for nums (median)
        # NO statistics.median(self.vals)
        listLen = len(self.vals)
        self.vals.sort()

        if listLen == 0:
            self.median = 0
            return self.median

        if listLen % 2 == 0:
            median1 = self.vals[listLen//2]
            median2 = self.vals[listLen//2 - 1]
            median = (median1 + median2)/2
        else:
            median = self.vals[listLen//2]

        self.median = median

        return self.median
         #returns median

    def dist(self, x, y): #Aha's distance bw two nums
        if (x == "?" or x == "") and (y == "?" or y == ""):
            return 1
        if (x == "?" or x == "") or (y == "?" or y == ""):
            x = x if (y == "?" or y == "") else y
            x = self._numNorm(x)
            y = 0 if x > 0.5 else 1
            return y - x
        return abs(self._numNorm(x) - self._numNorm(y))

    def _numNorm(self, x):
        "normalize the column." #Min-Max Normalization + a super tiny num so you never divide by 0
        return (x - self.lo)/(self.hi - self.lo + tiny)


# ------------------------------------------------------------------------------
# Table Class: Reads in CSV and Produces Table of Nums/Syms
# ------------------------------------------------------------------------------
class Table:
    def __init__(self, uid):
        self.uid = uid
        self.count = 0
        self.cols = []
        self.rows = []
        self.encodedrows = []
        self.fileline = 0
        self.linesize = 0
        self.skip = []
        self.y = []
        self.nums = []
        self.syms = []
        self.goals = []
        self.klass = []
        self.w = defaultdict(int)
        self.x = []
        self.xnums = [] #num x points (not including goals/klass)
        self.xsyms = [] #sym x points
        self.header = ""
        self.clabels = []

# ------------------------------------------------------------------------------
# Table Class: Helper Functions
# ------------------------------------------------------------------------------
    @staticmethod #helper function; not to create or instantiate; don't use with instantiated obj
    def compiler(x): # checks & compiles data type; a python thing
        try:
            int(x)
            return int(x)
        except:
            try:
                float(x)
                return float(x)
            except ValueError:
                return str(x)


    @staticmethod
    def readfile(file, sep= ",", doomed= r'([\n\t\r"\' ]|#.*)'): #reads in file
        datalines = []
        finallines = []

        with open(file) as f: #ensures that the file will be closed when control leaves the block
            curline = ""
            for line in f:
                line = line.strip() #get rid of all the white space
                if line[len(line) -1] == ",":
                    curline += line
                else:
                    curline += line
                    datalines.append(Table.compiler(curline))
                    curline = ""

        for l in datalines:
            line = l.strip()
            line = re.sub(doomed, '', line) # uses regular exp package to replace substrings in strings
            if line:
                finallines.append([Table.compiler(x) for x in line.split(sep)])
        return finallines  #returns all the pretty readable lines

# ------------------------------------------------------------------------------
# Table Class: Class Methods
# ------------------------------------------------------------------------------
    def __add__(self, line):
        if len(self.header) > 0: #if line has a header
            self.insert_row(line) # insert the row
        else:
            self.create_cols(line) #if not; create a col

    def create_cols(self, line):
        self.header = line #since add recognized no header, assign first line as a header.
        index = 0

        for val in line:
            val = self.compiler(val) #compile the val datatype

            if val[0] == ":" or val[0] == "?" : #do we skip? if we skip then it doesn't matter what we do? bc it'll never be populated?
                if val[0].isupper():
                    self.skip.append(Num(''.join(c for c in val), index)) # take all the items in val as long as it's not ?/: ;join()takes all items in an iterable and joins them as a string
                else:
                    self.skip.append(Sym(''.join(c for c in val), index))

            if val[0].isupper(): # is it a num?
                col = Num(''.join(c for c in val if not c in ['?',':']), index)
                self.nums.append(col)
                self.cols.append(col)

                if "!" in val or "-" in val or "+" in val: #is it a klass, or goal (goals are y)
                    self.y.append(col)
                    self.goals.append(col)
                    if "-" in val:
                        self.w[index] = -1
                    if "+" in val:
                        self.w[index] = 1
                    if "!" in val:
                        self.klass.append(col)

                if "-" not in val and "+" not in val and "!" not in val: # then it's an x
                    self.x.append(col)
                    self.xnums.append(col)

            else: #no, it's a sym
                col = Sym(''.join(c for c in val if not c in ['?',':']), index)
                self.syms.append(col)
                self.cols.append(col)

                if "!" in val or "-" in val or "+" in val: #is it a klass, or goal (goals are y)
                    self.y.append(col)
                    self.goals.append(col)
                    if "-" in val:
                        self.w[index] = -1
                    if "+" in val:
                        self.w[index] = 1
                    if "!" in val:
                        self.klass.append(col)

                if "-" not in val and "+" not in val and "!" not in val:
                    self.x.append(col)
                    self.xnums.append(col)

            index+=1 #increase by one
            self.linesize = index
            self.fileline += 1


    def insert_row(self, line):
        self.fileline +=1
        if len(line) != self.linesize:
            print("len(line)" , len(line), "self.linesize", self.linesize)
            print("Line", self.fileline, "has an error")
            return

        if isValid(self, line):
            realline = []
            index = 0
            for val in line:
                if index not in self.skip: #check if it needs to be skipped
                    if val == "?" or val == "":
                        realline.append(val) #add to realline
                        index += 1
                        continue
                    self.cols[index] + self.compiler(val)
                    realline.append(val)
                index += 1

            self.rows.append(realline)
            self.count += 1
        # else:
            # print("Line", self.fileline, "has missing values")

    def encode_lines(self):
        # for all Syms
        # initialize the LabelEncoder
        # fit from all dictionary keys
        encodedrows = []
        for col in self.cols:
            if col.coltype == 0:
                keys = list(col.count.keys())
                col.encoder.fit(keys)
        for line in self.rows:
            newline = []
            for i, val in enumerate(line):
                newval = val
                if self.cols[i].coltype == 0:
                    newval = self.cols[i].encoder.transform([val])[-1]
                else:
                    newval = self.compiler(val)
                newline.append(newval)
            encodedrows.append(newline)
        self.encodedrows = encodedrows
        # for all lines, if col of line is Sym encode with le.transform([val])
        # store all encoded lines

# ------------------------------------------------------------------------------
# Clustering Fastmap;still in table class (change to it's own class???)
# ------------------------------------------------------------------------------
    def split(self, top = None):#Implements continous space Fastmap for bin chop on data
        if top == None:
            top = self
        pivot = random.choice(self.rows) #pick a random row
        #import pdb;pdb.set_trace()
        left = top.mostDistant(pivot, self.rows) #get most distant point from the pivot
        right = top.mostDistant(left, self.rows) #get most distant point from the leftTable
        c = top.distance(left,right) #get distance between two points
        items = [[row, 0] for row in self.rows] #make an array for the row & distance but initialize to 0 to start

        for x in items:
            a = top.distance(x[0], right) # for each row get the distance between that and the farthest point right
            b = top.distance(x[0], left) # for each row get the distance between that and the farthest point left
            x[1] = (a ** 2 + c**2 - b**2)/(2*c + 10e-32) #cosine rule for the distance assign to dist in (row, dist)

        items.sort(key = lambda x: x[1]) #sort by distance (method sorts the list ascending by default; can have sorting criteria)
        splitpoint = len(items) // 2 #integral divison
        leftItems = [x[0] for x in items[: splitpoint]] #left are the rows to the splitpoint
        rightItems = [x[0] for x in items[splitpoint :]] #right are the rows from the splitpoint onward

        return [top, left, right, leftItems, rightItems]

    def distance(self,rowA, rowB): #distance between two points
        distance = 0
        if len(rowA) != len(rowB): #catch if they can't be compared?? why??
            return -big
        # for i, (a,b) in enumerate(zip(rowA, rowB)):#to iterate through an interable: an get the index with enumerate(), and get the elements of multiple iterables with zip()
        for col in self.cols: #to include y self.cols ; for just x vals self.x
            i = col.uid
            d = self.cols[i].dist(self.compiler(rowA[i]),self.compiler(rowB[i])) #distance of both rows in each of the columns; compile the a & b bc it's in a text format
            distance += d #add the distances together
        return distance

    def mostDistant(self, rowA, localRows): #find the furthest point from row A
        distance = -big
        farthestRow = None # assign to null; python uses None datatype

        for row in self.rows:
            d = self.distance(rowA, row) #for each of the rows find the distance to row A
            if d > distance: #if it's bigger than the distance
                distance = d #assign the new distance to be d
                farthestRow = row #make point the far row
        #print("most distant = ", distance, "away and is ", farthestRow[-1])
        return farthestRow #return the far point/row

    def closestPoint(self,rowA):
        distance = big
        closestRow = None # assign to null; python uses None datatype
        secondClosest = None

        for row in self.rows:
            d = self.distance(rowA, row) #for each of the rows find the distance to row A
            if d < distance: #if it's smaller than the distance
                distance = d #assign the new distance to be d
                closestRow = row #make point the close row
        return closestRow #return the close point/row

    @staticmethod

    def clusters(items, table, enough, top = None, depth = 0):
        # print("|.. " * depth,len(table.rows))
        # print("top cluster:", top)
        if len(items) < enough: # if/while the length of the less than the stopping criteria #should be changable from command line
            leftTable = Table(0) #make a table w/ uid = 0
            leftTable + table.header # able the table header to the table ; leftTable.header = table.header?
            for item in items: #add all the items to the table
                leftTable + item
            return TreeNode(None, None, leftTable, None, table, None, None, True, table.header) #make a leaf treenode when the cluster have enough rows in them
        #if you don't enough items
        if top != None:
            _, left, right, leftItems, rightItems = table.split(top)
        else:
            top, left, right, leftItems, rightItems = table.split(top)

        leftTable = Table(0)
        leftTable + table.header
        for item in leftItems:
            leftTable + item

        rightTable = Table(0)
        rightTable + table.header
        for item in rightItems:
            rightTable + item
        # print(rightTable.rows)
        leftNode = Table.clusters(leftItems, leftTable, enough, top, depth = depth+1)
        rightNode = Table.clusters(rightItems, rightTable, enough, top, depth = depth+1)
        root = TreeNode(left, right, leftTable, rightTable, table, leftNode, rightNode, False, table.header)
        return root

# ------------------------------------------------------------------------------
# Tree class
# ------------------------------------------------------------------------------
class TreeNode:
    _ids = count(0)
    def __init__(self, left, right, leftTable, rightTable, currentTable, leftNode, rightNode, leaf, header):
        self.uid = next(self._ids)
        self.left = left
        self.right = right
        self.leftTable = leftTable
        self.rightTable = rightTable
        self.currentTable = currentTable
        self.leaf = leaf
        self.header = header
        self.leftNode = leftNode
        self.rightNode = rightNode

# ------------------------------------------------------------------------------
# TreeNode Class Helper Fuctions: Functional Tree Traversal
# ------------------------------------------------------------------------------

def nodes(root): # gets all the leaf nodes
    if root:
        for node in nodes(root.leftNode): yield node #yield returns from a function without destroying it
        if root.leaf:  yield root
        for node in nodes(root.rightNode): yield node

def names(root:TreeNode): #gets all the col names of the node
    for node in nodes(root):
        for i in range(len(node.leftTable.cols) -1):
            print(node.leftTable.cols[i].name)

def rowSize(t): return len(t.leftTable.rows) #gets the size of the rows

def leafmedians(root,how=None): # for all of the leaves from smallest to largest print len of rows & median
    MedianTable = Table(222)
    MedianTable.create_cols(root.header)
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        mid = [col.mid() for col in t.cols]
        MedianTable + mid
        #print(len(t.rows), [col.mid() for col in t.cols], t.cols[-1].count)
    MedianTable.encode_lines()
    return MedianTable

def getLeafData(root,samples_per_leaf,how=None): # for all of the leaves from smallest to largest print len of rows & median
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


def sortedleafclusterlabels(root,f,how=None): # for all of the leaves from smallest to largest print len of rows & median
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
                xlabel = str(row[len(row)-1])
                if xlabel == clabel: #this will crash if the xlabel is a string and the clabel is an int (i.e GermanCredit)
                    match += 1

        t.clabels = [clabel for i in range(len(t.rows))]
        matches = match/(len(t.rows))


        f.write("Leaf " + str(counter)+"\n")
        if matches >= 0.80:
            f.write("--------------------------------> Good Cluster Label <--------" +"\n")
        else:
            f.write("Bad Cluster Label" +"\n")

        percent = "{0:.0%}".format(matches, 2)
        f.write("Cluster Label: " + str(clabel) +"\n")
        f.write("Label Matches: " + str(match) + "/" + str(len(t.rows))+"\n")
        f.write("Label Matches Percentage: " + str(percent) +"\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------------------"+"\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------------------"+"\n")

        match = 0



    def dump(self, f):
        #DFS
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
        #DFS
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



# ------------------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------------------
# class Abcd:
#   def __init__(i,db="all",rx="all"):
#     i.db = db; i.rx=rx;
#     i.yes = i.no = 0
#     i.known = {}; i.a= {}; i.b= {}; i.c= {}; i.d= {}
#
#   def __call__(i,actual=None,predicted=None):
#     return i.keep(actual,predicted)
#
#   def tell(i,actual,predict):
#     i.knowns(actual)
#     i.knowns(predict)
#     if actual == predict: i.yes += 1
#     else                :  i.no += 1
#     for x in  i.known:
#       if actual == x:
#         if  predict == actual: i.d[x] += 1
#         else                 : i.b[x] += 1
#       else:
#         if  predict == x     : i.c[x] += 1
#         else                 : i.a[x] += 1
#
#   def knowns(i,x):
#     if not x in i.known:
#       i.known[x]= i.a[x]= i.b[x]= i.c[x]= i.d[x]= 0.0
#     i.known[x] += 1
#     if (i.known[x] == 1):
#       i.a[x] = i.yes + i.no
#
#   def header(i):
#     print("#",('{0:20s} {1:11s}  {2:4s}  {3:4s} {4:4s} '+ \
#            '{5:4s}{6:4s} {7:3s} {8:3s} {9:3s} '+ \
#            '{10:3s} {11:3s}{12:3s}{13:10s}').format(
#       "db", "rx",
#      "n", "a","b","c","d","acc","pd","pf","prec",
#       "f","g","class"))
#     print('-'*100)
#
#   def ask(i):
#     def p(y) : return int(100*y + 0.5)
#     def n(y) : return int(y)
#     pd = pf = pn = prec = g = f = acc = 0
#     for x in i.known:
#       a= i.a[x]; b= i.b[x]; c= i.c[x]; d= i.d[x]
#       if (b+d)    : pd   = d     / (b+d)
#       if (a+c)    : pf   = c     / (a+c)
#       if (a+c)    : pn   = (b+d) / (a+c)
#       if (c+d)    : prec = d     / (c+d)
#       if (1-pf+pd): g    = 2*(1-pf)*pd / (1-pf+pd)
#       if (prec+pd): f    = 2*prec*pd/(prec+pd)
#       if (i.yes + i.no): acc= i.yes/(i.yes+i.no)
#       print("#",('{0:20s} {1:10s} {2:4d} {3:4d} {4:4d} '+ \
#           '{5:4d} {6:4d} {7:4d} {8:3d} {9:3d} '+ \
#          '{10:3d} {11:3d} {12:3d} {13:10s}').format(i.db,
#           i.rx,  n(b + d), n(a), n(b),n(c), n(d),
#           p(acc), p(pd), p(pf), p(prec), p(f), p(g),x))
#       #print x,p(pd),p(prec)


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
def test_num(): #copied Dr.M's test
    test_data = []
    for i in range(100):
        test_data.append(random.random())
    n = Num(11, 11, data=test_data)
    #print("n.diversity():", str(n.diversity()))
    assert .25 <= n.diversity() <= .31, "in range" #changed range check (confirm is this is okay to do)
    # assert n.dist(), "num distance"

def test_sym():
    test_data = ["a","a","a","a","b","b","c"]
    s = Sym(12,12, data = test_data)
    #print("s.diversity():", str(s.diversity()))
    assert 1.37 <= s.diversity() <= 1.38, "entropy"
    assert 'a'  == s.mid(), "mode"
    assert 0 == s.dist('a','a'), "same sym distance"
    assert 1 == s.dist('a','b'), "diff sym distance"

def test_rows():
    lines = Table.readfile("test.csv")
    table = Table(0)
    ls = table.linemaker(lines)
    for line in ls:
        table + line
    num_rows = len(lines)
    #print("num_rows:", num_rows) #count includes the header
    assert 101 == num_rows, "counting rows"



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

def classify(table):
    X = []
    y = []
    all_data ={}

    y_index = table.y[-1].uid
    for row in table.encodedrows:
        X_row = []
        y_row = -1
        for i, val in enumerate(row):
            if i == y_index: #for multiple y if i in y_indexes:
                y_row=val
            else:
                X_row.append(val)
        X.append(X_row)
        y.append(y_row)

    # create svm
    svclassifier = SVC(kernel='linear')
    # clf = RandomForestClassifier(random_state=0)

    # predict
    for i in range(1,11): # split data 80% train 20 % test * 10 times
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)

        # evaluation
        # cm = confusion_matrix(y_test,y_pred)
        # print(cm)
        cr = classification_report(y_test,y_pred, output_dict = True)
        # print("i:", i, "cr:", cr)
        cr_data = []
        cr_data.append(len(table.rows))
        cr_data.append(cr['accuracy'])
        # cr_data.append(cr['macro avg']['support'])
        cr_data.append(cr['0']['precision'])
        cr_data.append(cr['0']['recall'])
        cr_data.append(cr['0']['f1-score'])

        try:
            cr_data.append(cr['1']['precision'])
            cr_data.append(cr['1']['recall'])
            cr_data.append(cr['1']['f1-score'])
        except:
            print("Theres no positive class for this cr...")
            cr_data.append(0)
            cr_data.append(0)
            cr_data.append(0)


        cr_data.append(cr['macro avg']['precision'])
        cr_data.append(cr['macro avg']['recall'])
        cr_data.append(cr['macro avg']['f1-score'])
        # cr_data.append(cr['weighted avg']['precision'])
        # cr_data.append(cr['weighted avg']['recall'])
        # cr_data.append(cr['weighted avg']['f1-score'])


        all_data[i] = cr_data
    # print(cr)

    return all_data

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def clusterandclassify(csv, limiter = None):
    dataset = csv
    filename = dataset[:-4] #cut off the .csv
    # colnames = ['accuracy', 'support', 'precision', 'recall', 'f1-score', 'precision_weighted', 'recall_weighted', 'f1-score_weighted', 'samples']
    colnames = ['samples','accuracy', 'class0_precision', 'class0_recall',
    'class0_f1-score','class1_precision', 'class1_recall', 'class1_f1-score', 'macro_precision', 'macro_recall', 'macro_f1-score']
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
    print(list(table.y[-1].encoder.classes_))
    data[len(table.rows)] = classify(table)

    # print("Clustering ...")
    enough = int(math.sqrt(len(table.rows)))
    root = Table.clusters(table.rows, table, enough)

    # print("Sorting leaves ...")
    # print("Extrapolated Data Classification... until", (enough//2), "samples")
    # leafmedians(root) #bfs for the leaves gives median row
    pbar = tqdm(list(range(1,(int(enough*0.75))))) #loading bar
    for samples in pbar:
        pbar.set_description("Extrapolated Data Classification with %s samples" % samples)
        MedianTable = leafmedians(root)
        # print("MT rows:", MedianTable.rows)
        data[samples] = classify(MedianTable)
        # EDT = getLeafData(root, samples) #get one random point from leaves
        # data[samples] = classify(EDT)

    for key, v in data.items():
        # print("data dict: " , data)
        for key2 in v.items():
            # print("key2", key2)
            tmp = key2[1]
            x_data.append(tmp)

    # print("x_data ", x_data)
    df = pd.DataFrame(x_data, columns=colnames)
    # print(df.head())

    df.to_csv("./output/"+filename + "_median_SVM.csv", index=False)


def main():
    random.seed(10019)
    datasets = ["CleanCOMPAS53.csv", "GermanCredit.csv"]
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        clusterandclassify(dataset, limiter = 1000)
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
