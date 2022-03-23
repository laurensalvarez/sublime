from collections import defaultdict
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
        self.uid = Col.id(self) #uid --> it allows for permanence and recalling necessary subtables
        self.count = defaultdict(int) #will never throw a key error bc it will guve default value as missing key
        if data != None: #initializes the empty col with val
            for val in data:
                self + val #calls __add__

    #def __str__(self):
        #print to a sym; overrides print(); could replace dump() TBD

    def __add__(self, v): return self.add(v,1) #need to adds? i forgot why

    def add (self, v, inc=1): #want to be able to control the increments
        self.n += inc # add value to the count
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
        # print("Aha's SYM ...x", x)
        # print("Aha's SYM ...y", y)
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
        self.hi = -tiny #-float('inf')
        self.vals = []
        self.uid = uid
        self.median = 0
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
            print("failed col name:", self.name, self.uid)
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
        self.vals.sort()
        listLen = len(self.vals)
        m = listLen//2
        if listLen% 2 == 0:
            m = (self.vals[m-1]+self.vals[m])/2
            return self.median
        else:
            self.median = self.vals[m]
            return self.median
         #returns median


    def dist(self, x, y): #Aha's distance bw two nums
        # print("Aha's Nums ...x", x)
        # print("Aha's Nums ...y", y)
        if (x == "?" or x == "") and (y == "?" or y == ""):
            return 1
        if (x == "?" or x == "") or (y == "?" or y == ""):
            x = x if (y == "?" or y == "") else y
            x = self._numNorm(x)
            y = 0 if x > 0.5 else 1
            return y - x
        return self._numNorm(x) - self._numNorm(y)

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


    def fillblanks(line):
        linenoblanks = ""


        return linenoblanks
    @staticmethod
    def readfile(file, sep= ",", doomed= r'([\n\t\r ]|#.*)'): #reads in file
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
            print("Line", self.fileline, "has an error")
            return
        realline = []
        realindex = 0
        index = 0
        for val in line:
            if index not in self.skip: #check if it needs to be skipped
                if val == "?" or val == "":
                    realline.append(val) #add to realline index
                    realindex += 1
                    continue
                self.cols[realindex] + self.compiler(val)
                realline.append(val)
                realindex += 1
            else: #otherwise add the skipped row too
                realindex += 1
            index += 1
        self.rows.append(line)
        self.count += 1

# ------------------------------------------------------------------------------
# TODO: replace how this prints with new list construction
# ------------------------------------------------------------------------------
    def dump(self, f):
        f.write("Dump table:"+"\n")
        f.write("table.cols stats info"+"\n")
        for i, col in enumerate(self.cols):
            if i in self.skip:
                continue
            if i in self.nums:
                f.write("|  " + "we're looking at col #" +str(col.uid)+"\n")
                f.write("|  |  col:  "+str(col.uid)+"\n")
                f.write("|  |  hi:   "+str(col.hi)+"\n")
                f.write("|  |  lo:   "+str(col.lo)+"\n")
                f.write("|  |  m2:   "+str(col.m2)+"\n")
                f.write("|  |  mu:   "+str(col.mu)+"\n")
                f.write("|  |  n:    "+str(col.n)+"\n")
                f.write("|  |  sd:   "+str(col.sd)+"\n")
                f.write("|  |  name: "+str(col.name)+"\n")
            else:
                f.write("|  " + str(col.uid) + "\n")
                f.write("|  |  col:  "+str(col.uid)+"\n")
                f.write("|  |  mode: "+str(col.mode)+"\n")
                f.write("|  |  most: "+str(col.most)+"\n")
                f.write("|  |  n:    " + str(col.n) + "\n")
                f.write("|  |  name: " + str(col.name) + "\n")

        f.write("table x & y info: "+"\n")
        f.write("|  len(cols): " + str(len(self.cols))+"\n")
        f.write("|  y" + "\n")
        for v in self.y:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  nums" + "\n")
        for v in self.nums:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  syms" + "\n")
        for v in self.syms:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  w" + "\n")
        for k, v in self.w.items():
            if v not in self.skip:
                f.write("|  |  " + str(k) + ": "+str(v)+"\n")
        f.write("|  x" + "\n")
        for v in self.x:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  xnums" + "\n")
        for v in self.xnums:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  xsyms" + "\n")
        for v in self.xsyms:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")

    def ydump(self, f):
        f.write("how many table cols: " + str(len(self.cols))+"\n")
        f.write("leaf table's y col info: "+"\n")
        for v in self.y:
            if v not in self.skip:
                f.write("y index: " + str(v) + "\n")
        for i, col in enumerate(self.cols):
            if i in self.y:
                if i in self.skip:
                    continue
                if i in self.nums:
                    f.write("|  " + "we're looking at NUM col id #" +str(col.uid)+"\n")
                    f.write("|  " + "we're looking at y index" +str(i)+"\n")
                    f.write("|  |  n:    "+str(col.n)+"\n")
                    f.write("|  |  median:    "+str(col.median)+"\n")
                    f.write("|  |  col:  "+str(col.vals)+"\n")
                    f.write("|  |  name: "+str(col.name)+"\n")
                else:
                    f.write("| SYM col id # " + str(col.uid) + "\n")
                    f.write("|  |  mode: "+str(col.mode)+"\n")
                    f.write("|  |  most: "+str(col.most)+"\n")
                    for k, v in col.count.items():
                        f.write("|  |  SYM Key : Value --> " + str(k) + ": " + str(v) + "\n")
    def xdump(self, f):
        f.write("how many table cols: " + str(len(self.cols))+"\n")
        f.write("leaf table's x col info: "+"\n")
        f.write("x indexes: " + str(len(self.x)) + "\n")
        for v in self.rows:
            if v not in self.skip:
                f.write("row class: " + str(v[len(v)-1]) + "\n")
        for i, col in enumerate(self.cols):
            if i in self.x:
                if i in self.skip:
                    continue
                if i in self.nums:
                    f.write("|  " + "we're looking at NUM col id #" +str(col.uid)+"\n")
                    # f.write("|  |  n:    "+str(col.n)+"\n")
                    f.write("|  |  median:    "+str(col.median)+"\n")
                    # f.write("|  |  col:  "+str(col.vals)+"\n")
                    f.write("|  |  name: "+str(col.name)+"\n")
                else:
                    f.write("| SYM col id # " + str(col.uid) + "\n")
                    f.write("|  |  mode: "+str(col.mode)+"\n")
                    f.write("|  |  most: "+str(col.most)+"\n")
                    f.write("|  |  name: "+str(col.name)+"\n")
                    for k, v in col.count.items():
                        f.write("|  |  SYM Key : Value --> " + str(k) + ": " + str(v) + "\n")


# ------------------------------------------------------------------------------
# Clustering Fastmap;still in table class (change to it's own class???)
# ------------------------------------------------------------------------------
    def split(self, left = None, right = None):#Implements continous space Fastmap for bin chop on data
    #instead of keeping top cluster, kept the top's left and right points then ask top cluster's left & right what's the most distant point in the local cluster
        # top = top or self
        if left == None and right == None:
            pivot = random.choice(self.rows) #pick a random row
            #import pdb;pdb.set_trace()
            left = self.mostDistant(pivot) #get most distant point from the pivot
            right = self.mostDistant(left) #get most distant point from the leftTable
        c = self.distance(left,right) #get distance between two points
        items = [[row, 0] for row in self.rows] #make an array for the row & distance but initialize to 0 to start

        for x in items:
            a = self.distance(x[0], right) # for each row get the distance between that and the farthest point right
            b = self.distance(x[0], left) # for each row get the distance between that and the farthest point left
            x[1] = (a ** 2 + c**2 - b**2)/(2*c + 10e-32) #cosine rule for the distance assign to dist in (row, dist)

        items.sort(key = lambda x: x[1]) #sort by distance (method sorts the list ascending by default; can have sorting criteria)
        splitpoint = len(items) // 2 #integral divison
        leftItems = self.rows[: splitpoint] #left are the rows to the splitpoint
        rightItems = self.rows[splitpoint :] #right are the rows from the splitpoint onward

        return [left, right, leftItems, rightItems]

    def distance(self,rowA, rowB): #distance between two points
        distance = 0
        if len(rowA) != len(rowB): #catch if they can't be compared?? why??
            return -tiny
        for i, (a,b) in enumerate(zip(rowA, rowB)): #to iterate through an interable: an get the index with enumerate(), and get the elements of multiple iterables with zip()
            d = self.cols[i].dist(self.compiler(a),self.compiler(b)) #distance of both rows in each of the columns; compile the a & b bc it's in a text format
            distance += d #add the distances together
        return distance

    def mostDistant(self, rowA): #find the furthest point from row A
        distance = -tiny
        farthestRow = None # assign to null; python uses None datatype

        for row in self.rows:
            d = self.distance(rowA, row) #for each of the rows find the distance to row A
            if d > distance: #if it's bigger than the distance
                distance = d #assign the new distance to be d
                farthestRow = row #make point the far row
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
    def clusters(items, table, enough, left = None, right= None, depth = 0):
        print("|.. " * depth,len(table.rows))
        if len(items) < enough: # if/while the length of the less than the stopping criteria #should be changable from command line
            leftTable = Table(0) #make a table w/ uid = 0
            leftTable + table.header # able the table header to the table ; leftTable.header = table.header?
            for item in items: #add all the items to the table
                leftTable + item
            return TreeNode(None, None, leftTable, None, table, None, None, True, table.header) #make a leaf treenode when the cluster have enough rows in them
        #if you don't enough items
        if left != None and right != None:
            _, _, leftItems, rightItems = table.split(left, right) #fastmap bin split on the table
        else:
            left, right, leftItems, rightItems = table.split(left, right)

        leftTable = Table(0)
        leftTable + table.header
        for item in leftItems:
            leftTable + item

        rightTable = Table(0)
        rightTable + table.header
        for item in rightItems:
            rightTable + item
        # print(rightTable.rows)
        leftNode = Table.clusters(leftItems, leftTable, enough, left, right, depth = depth+1)
        rightNode = Table.clusters(rightItems, rightTable, enough, left, right, depth = depth+1)
        root = TreeNode(left, right, leftTable, rightTable, table, leftNode, rightNode, False, table.header)
        return root

# ------------------------------------------------------------------------------
# TODO: replace how this prints with new list construction
# ------------------------------------------------------------------------------
    def csvDump(self, f):
        for i, col in enumerate(self.cols):
            if i in self.skip:
                continue
            if i in self.nums:
                f.write(str(col.uid) + ",")
                f.write(str(col.hi)+",")
                f.write(str(col.lo)+",")
                f.write(str(col.m2)+",")
                f.write(str(col.mu)+",")
                f.write(str(col.n)+",")
                f.write(str(col.sd)+",")
            else:
                f.write(str(col.uid)+",")
                f.write(str(col.mode)+",")
                f.write(str(col.most)+",")
                f.write(str(col.n) + ",")
        f.write("\n")

    def csvHeader(self):
        header = ""
        for i, col in enumerate(self.cols):
            if i in self.skip:
                continue
            if i in self.nums:
                header += (str(col.name)+"_uid,")
                header += (str(col.name)+"_hi,")
                header += (str(col.name)+"_lo,")
                header += (str(col.name)+"_m2,")
                header += (str(col.name)+"_mu,")
                header += (str(col.name)+"_n,")
                header += (str(col.name)+"_sd,")
            else:
                header += (str(col.name)+"_uid,")
                header += (str(col.name)+"_mode,")
                header += (str(col.name)+"_most,")
                header += (str(col.name)+"_n,")
        header += "\n"
        return header


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

def nodes(root): # gets all the  nodes
    if root:
        for node in nodes(root.leftNode): yield node #yield returns from a function without destroying it
        if root.leaf:  yield root
        for node in nodes(root.rightNode): yield node

def names(root:TreeNode): #gets all the col names of the node
    for node in nodes(root):
        for i in range(len(node.leftTable.cols) -1):
            print(node.leftTable.cols[i].name)

def rowSize(t): return len(t.leftTable.rows) #gets the size of the rows

def small2Big(root,how=None): # for all of the leaves from smallest to largest print len of rows & median
  for leaf in sorted(nodes(root), key=how or rowSize):
    t = leaf.leftTable
    print(len(t.rows), [col.mid() for col in t.cols])

def sortedleafclusterlabels(root,f,how=None): # for all of the leaves from smallest to largest print len of rows & median
    clabel = None
    xlabel = None
    match = 0
    counter = 0

    for leaf in sorted(nodes(root), key=how or rowSize):
        counter += 1
        t = leaf.leftTable

        clabel = t.y[0].mid()
        # print("t.y:", str(t.y))
        # print("t.y mid():", t.y[0].mid())

        for row in t.rows:
            if row not in t.skip:
                xlabel = str(row[len(row)-1])
                if xlabel == clabel:
                    match += 1

        t.clabels = [clabel for i in range(len(t.rows))]
        matches = match/(len(t.rows)-1)

        f.write("Leaf " + str(counter)+"\n")
        if matches >= 0.80:
            f.write("--------------------------------> Good Cluster Label <--------" +"\n")
        else:
            f.write("Bad Cluster Label" +"\n")

        percent = "{0:.0%}".format(match/(len(t.rows)-1), 2)
        f.write("Cluster Label: " + str(clabel) +"\n")
        f.write("Label Matches: " + str(match) + "/" + str(len(t.rows)-1)+"\n")
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
# Main
# ------------------------------------------------------------------------------
def datasetswitch(csv):
    dataset = csv
    filename = dataset[:-4] #cut off the .csv

    print("---------------------------")
    print("DS:", str(filename))
    print("---------------------------")

    lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/" + dataset)
    table = Table(1)
    table + lines[0]
    for l in lines[1:]:
        table + l
    print("CSV --> Table done ...")

    print("Shuffling rows ...")
    random.shuffle(table.rows)

    print("Clustering ...")
    root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))

    print("Sorting leaves ...")
    small2Big(root) #bfs for the leaves gives median row

    print("Comparing cluster labels to ground truths ...")
    with open( filename + "_BFS.csv", "w") as f:
        sortedleafclusterlabels(root,f)


    print("Performance Metrics ...")
    # abcd = Abcd(db='randomIn',rx='all')
    # train = table.clabels
    # test  = table.y

    # # random.shuffle(test)
    # for actual, predicted in zip(train,test):
    #     abcd.tell(actual,predicted)
    # abcd.header()
    # abcd.ask()

    print("---------------------------")
    print("--- completed")
    print("---------------------------")



def main():
    # test_num()
    # test_sym()
    # test_rows()
    # print("---------------------------")
    # print("All 3 unit tests passed")
    # print("---------------------------")
    # print("---------------------------")

    # print("---------------------------")
    # print("DS 1: Diabetes Case:")
    # print("---------------------------")
    #
    # lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/diabetes.csv")
    # table = Table(1)
    # table + lines[0]
    # for l in lines[1:]:
    #     table + l
    # print("CSV --> Table done ...")
    # # print("---------------------------")
    # # print("Printing all attributes ")
    # # print(" Table Cols:", table.cols)
    # # print(" Table Num Cols Vals:", table.cols[0].vals)
    # # print(" Table Num Cols Mid:", table.cols[0].median)
    # # print(" Table Rows:", len(table.rows))
    # # print(" Table Skips:", len(table.skip))
    # # print(" Table Goals:", len(table.goals))
    # # print(" Table Klass:", len(table.klass))
    # # print(" Table Header:", table.header)
    # # print(" Table Nums:", len(table.nums))
    # # print(" Table Syms:", len(table.syms))
    # # print(" Table xNums:", table.xnums)
    # # print(" Table xSyms:", table.xsyms) #why is this 2???
    # # ##########################
    # print("Shuffling rows ...")
    # random.shuffle(table.rows)
    # print("Clustering ...")
    # root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    # small2Big(root) #bfs for the leaves gives median row
    # with open("FR_diabetes_BFS.csv", "w") as f:
    #     sortedleafclusterlabels(root,f)

    # abcd = Abcd(db='randomIn',rx='all')
    # train = table.clabels
    # test  = table.y
    # print("how many cluster labels:", len(table.clabels))
    # print("how many test/y labels:", len(table.y))
    # # random.shuffle(test)
    # for actual, predicted in zip(train,test):
    #     abcd.tell(actual,predicted)
    # abcd.header()
    # abcd.ask()
    # print("---------------------------")
    # print("--- completed")
    # print("---------------------------")

    print("---------------------------")
    print("Other Datasets:")
    print("---------------------------")
    datasetswitch("diabetes.csv")
    # datasetswitch("adultscensusincome.csv")
    # datasetswitch("bankmarketing.csv")
    # datasetswitch("COMPAS53.csv")
    # datasetswitch("GermanCredit.csv")
    # datasetswitch("processed.clevelandhearthealth.csv")
    # datasetswitch("defaultcredit.csv")
    # datasetswitch("homecreditapplication_train.csv") # loaded 266113 rows after 2 hours; error on compiling sym/num cols



    # print("---------------------------")
    # print("DS 7: Home Credit Case:")
    # print("---------------------------")
    #



if __name__ == '__main__':
    main()
