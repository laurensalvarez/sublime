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
        if (x == "?" or x == "") or (y == "?" or y == ""): #check if the empty is just a bug
            return 1
        return 0 if x == y else 1

# ------------------------------------------------------------------------------
# Numeric Column Class
# ------------------------------------------------------------------------------
#big = sys.maxsize
#tiny = 1/big
class Num(Col):
    def __init__(self, name, uid, data=None):
        Col.__init__(self, name)
        self.n = 0
        self.mu = 0 #
        self.m2 = 0 # for moving std dev
        self.sd = 0
        self.lo = sys.maxsize #float('inf')
        self.hi = -1/sys.maxsize #-float('inf')
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
        return (x - self.lo)/(self.hi - self.lo + 10e-32)


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
    def readfile(file): #reads in file
        lines = [] #create a list of lines
        with open(file) as f: #ensures that the file will be closed when control leaves the block
            curline = "" #current line is an empty string to start
            for line in f:
                line = line.strip() #get rid of all the white space
                if line[len(line) -1] ==",": #if you reach a comma go to the next line
                    curline += line #add line to current
                else:
                    curline += line #add line to current
                    lines.append(curline) #add currentline to list
                    curline = "" #assign curline back to empty
        return lines #return a list of lines

    @staticmethod
    def linemaker(src, sep=",", doomed=r'([\n\t\r ]|#.*)'): #creates readable lines
        lines = [] #create a list of lines
        for line in src:
            line = line.strip() #removes any spaces or specified characters at the start and end of a string
            line = re.sub(doomed, '', line) # uses regular exp package to replace substrings in strings
            if line:
                lines.append(line.split(sep)) #put the good string back together
        return lines #returns all the pretty readable lines
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
            if val[0] == ":": #check the first item is : then skip it; add to skip list
                self.skip.append(index) # index begins with 1
            if val[0].isupper() or "-" in val or "+" in val: #assuming goals will be numeric cols
                self.nums.append(index) # add to num
                self.cols.append(Num(''.join(c for c in val if not c in ['?',':']), index)) # take all the items in val as long as it's not ?/: ;join()takes all items in an iterable and joins them as a string
            else:
                self.syms.append(index)
                self.cols.append(Sym(''.join(c for c in val if not c in ['?',":"]), index))

            if "!" in val or "-" in val or "+" in val: #for any goal, or klass add to y
                self.y.append(index)
                self.goals.append(index)
                if "-" in val:
                    self.w[index] = -1
                if "+" in val:
                    self.w[index] = 1
                if "!" in val:
                    self.klass.append(index)

            if "-" not in val and "+" not in val and "!" not in val: #catch the rest and add to x
                self.x.append(index)
                if val[0].isupper(): #check is num
                    self.xnums.append(index) #add the index of the col to the list
                else: #else add to sym
                    self.xsyms.append(index) #add the index of the col to the list
            index+=1 #increase by one
            self.linesize = index
            self.fileline += 1

    def insert_row(self, line):
        self.fileline +=1
        # print("inserting row", self.fileline, "of size", len(line), "expected = ", self.linesize)
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
            else: #otherwise add it to the rows and increase the count
                realindex += 1
            index += 1
        self.rows.append(line)
        self.count += 1

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
                # for k, v in col.cnt.items():
                #     f.write("|  |  |  " + str(k) + ": " + str(v) + "\n")
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

# ------------------------------------------------------------------------------
# Clustering Fastmap;still in table class (change to it's own class???)
# ------------------------------------------------------------------------------
    def split(self):#Implements continous space Fastmap for bin chop on data
        pivot = random.choice(self.rows) #pick a random row
        #import pdb;pdb.set_trace()
        east = self.mostDistant(pivot) #get most distant point from the pivot
        west = self.mostDistant(east) #get most distant point from the eastTable
        c = self.distance(east,west) #get distance between two points
        items = [[row, 0] for row in self.rows] #make an array for the row & distance but initialize to 0 to start

        for x in items:
            a = self.distance(x[0], west) # for each row get the distance between that and the farthest point west
            b = self.distance(x[0], east) # for each row get the distance between that and the farthest point east
            x[1] = (a ** 2 + c**2 - b**2)/(2*c + 10e-32) #cosine rule for the distance assign to dist in (row, dist)

        items.sort(key = lambda x: x[1]) #sort by distance (method sorts the list ascending by default; can have sorting criteria)
        splitpoint = len(items) // 2 #integral divison
        eastItems = self.rows[: splitpoint] #east are the rows to the splitpoint
        westItems = self.rows[splitpoint :] #west are the rows from the splitpoint onward

        return [east, west, eastItems, westItems]

    def distance(self,og_rowA, rowB): #distance between two points
        rowA = og_rowA
        distance = 0
        if len(rowA) != len(rowB): #catch if they can't be compared?? why??
            return -1/sys.maxsize
        for i, (a,b) in enumerate(zip(rowA, rowB)): #to iterate through an interable: an get the index with enumerate(), and get the elements of multiple iterables with zip()
            d = self.cols[i].dist(self.compiler(a),self.compiler(b)) #distance of both rows in each of the columns; compile the a & b bc it's in a text format
            distance += d #add the distances together
        return distance

    def mostDistant(self, og_rowA): #find the furthest point from row A
        rowA = og_rowA
        distance = -1/sys.maxsize
        farthestRow = None # assign to null; python uses None datatype

        for row in self.rows:
            d = self.distance(rowA, row) #for each of the rows find the distance to row A
            if d > distance: #if it's bigger than the distance
                distance = d #assign the new distance to be d
                farthestRow = row #make point the far row
        return farthestRow #return the far point/row

    @staticmethod
    def sneakClusters(items, table, enough):
        if len(items) < enough: # if/while the length of the less than the stopping criteria #should be changable from command line
            eastTable = Table(0) #make a table w/ uid = 0
            eastTable + table.header # able the table header to the table ; eastTable.header = table.header?
            for item in items: #add all the items to the table
                eastTable + item
            return TreeNode(None, None, eastTable, None, None, None, True, table.header) #make a treenode for east table
        #after you get enough items
        west, east, westItems, eastItems = table.split() #fastmap bin split on the table

        eastTable = Table(0)
        eastTable + table.header
        for item in eastItems:
            eastTable + item

        westTable = Table(0)
        westTable + table.header
        for item in westItems:
            westTable + item

        eastNode = Table.sneakClusters(eastItems, eastTable, enough)
        westNode = Table.sneakClusters(westItems, westTable, enough)
        root = TreeNode(east, west, eastTable, westTable, eastNode, westNode, False, table.header)
        return root #why do you return the root?


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
# Tree class for clustering
# ------------------------------------------------------------------------------
class TreeNode:
    _ids = count(0)
    def __init__(self, east, west, eastTable, westTable, eastNode, westNode, leaf, header):
        self.uid = next(self._ids)
        self.east = east
        self.west = west
        self.eastTable = eastTable
        self.westTable = westTable
        self.leaf = leaf
        self.header = header
        self.eastNode = eastNode
        self.westNode = westNode

    def dump(self, f):
        #DFS
        if self.leaf:
            f.write("Dump Leaf Node: " + str(self.uid) + "\n")
            f.write("Dump Leaf Table: " + "\n")
            self.eastTable.dump(f)
            return
        # f.write("Dump Node: " + str(self.uid) + "\n")
        # if self.eastTable is not None:
        #     f.write("Dump East Table: " + "\n")
        #     self.eastTable.dump(f)
        # else:
        #     f.write("No east table" + "\n")
        # if self.westTable is not None:
        #     f.write("Dump West Table: " + "\n")
        #     self.westTable.dump(f)
        # else:
        #     f.write("No west table" + "\n")
        if self.eastNode is not None:
        #     f.write("Dump East Node Recursive: " + "\n")
            self.eastNode.dump(f)
        # else:
        #     f.write("No east node" + "\n")
        if self.westNode is not None:
        #     f.write("Dump West Node Recursive: " + "\n")
            self.westNode.dump(f)
        # else:
        #     f.write("No west node" + "\n")

    def csvDump(self, f):
        #DFS
        if self.leaf:
            self.eastTable.csvDump(f)
            return

        if self.eastNode is not None:
            self.eastNode.csvDump(f)
        if self.westNode is not None:
            self.westNode.csvDump(f)

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

def main():
    # test_num()
    # test_sym()
    # test_rows()
    # print("---------------------------")
    # print("All 3 unit tests passed")
    # print("---------------------------")
    # print("---------------------------")
    # print("Small test case completed: can sort a dataset")
    # print("---------------------------")
    # ####### FULL TEST #########
    # lines = Table.readfile("test.csv")
    # table = Table(0)
    # ls = table.linemaker(lines)
    # for line in ls:
    #     table + line
    # print("---------------------------")
    # print("Testing Code with small fake data: test.csv ")
    # print("Test Table Cols:", table.cols)
    # print("Test Table Num Cols Vals:", table.cols[0].vals)
    # print("Test Table Sym Cols Dictionary:", table.cols[1].count)
    # print("Test Table Num Cols Mid:", table.cols[0].median)
    # print("Test Table Sym Cols Mid:", table.cols[1].mode)
    # print("Test Table Rows:", len(table.rows))
    # print("Test Table Skips:", len(table.skip))
    # print("Test Table Goals:", len(table.goals))
    # print("Test Table Klass:", len(table.klass))
    # print("Test Table Header:", table.header)
    # print("Test Table Nums:", len(table.nums))
    # print("Test Table Syms:", len(table.syms))
    # print("Test Table xNums:", table.xnums)
    # print("Test Table xSyms:", table.xsyms) #why is this 2???
    # ##########################
    # print("---------------------------")
    # print("---------------------------")
    # print("FIRST test completed")
    # print("---------------------------")

    print("---------------------------")
    print("Full Test Case: can cluster")
    print("---------------------------")

    lines = Table.readfile("auto93.csv")
    table = Table(1)
    ls = table.linemaker(lines)
    for line in ls:
        table + line
    root = Table.sneakClusters(table.rows, table, int(math.sqrt(len(table.rows))))

    with open("auto93_clusters.csv", "w") as f:
        csvheader = table.csvHeader()
        f.write(csvheader)
        root.csvDump(f)

    with open("auto93_clusters_desc.csv", "w") as f:
        csvheader = table.csvHeader()
        f.write(csvheader)
        root.dump(f)
    print("---------------------------")
    print("SECOND Test completed")
    print("---------------------------")


if __name__ == '__main__':
    main()
