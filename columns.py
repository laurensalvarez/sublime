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
        # print("self", self)
        # print("self.lo:", self.lo)
        # print("self.hi", self.hi)
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

    @staticmethod
    def readfile(file):
        lines = []
        with open(file) as f:
            curline = ""
            for line in f:
                line = line.strip()
                if line[len(line) -1] ==",":
                    curline += line
                else:
                    curline += line
                    lines.append(Table.compiler(curline))
                    curline = ""
        return lines

    @staticmethod
    def linemaker(src, sep=",", doomed=r'([\n\t\r ]|#.*)'):
        lines = []
        for line in src:
            line = line.strip()
            line = re.sub(doomed, '', line)
            if line:
                lines.append([Table.compiler(x) for x in line.split(sep)])
        return lines


    # @staticmethod
    # def readfile(file): #reads in file
    #     lines = []
    #     with open(file) as f: #ensures that the file will be closed when control leaves the block
    #         for line in f: #for all the lines in the file 1 by1
    #             line = line.strip() #get rid of all the white space
    #             lines.append(Table.compiler(line)) # add all the lines compiled
    #     # print("RETURN LINES:", lines)
    #     return lines #return a list of strings
    #
    # @staticmethod
    # def linemaker(src, sep=",", doomed=r'([\n\t\r ]|#.*)'):
    #     lines = [] #create a list of lines
    #     for line in src:
    #         line = line.strip() #removes any spaces or specified characters at the start and end of a string
    #         line = re.sub(doomed, '', line) # uses regular exp package to replace substrings in strings
    #         if line:
    #             lines.append([Table.compiler(x) for x in line.split(sep)]) #for every entry in the list of line elements add the complied
    #     return lines  #returns all the pretty readable lines

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
            if val[0] == ":" or val[0] == "?" : #check the first item is : then skip it; add to skip list
                self.skip.append(index) # index begins with 1
            if val[0].isupper() or "-" in val or "+" in val: #assuming goals will be numeric cols
                self.nums.append(index) # add to num
                # print("NUM col added")
                self.cols.append(Num(''.join(c for c in val if not c in ['?',':']), index)) # take all the items in val as long as it's not ?/: ;join()takes all items in an iterable and joins them as a string
            else:
                self.syms.append(index)
                # print("SYM col added")
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
            # print("LINE:" , line)
            # print("VAL in line:" , val)
            if index not in self.skip: #check if it needs to be skipped
                if val == "?" or val == "":
                    #val = self.compiler(val) #compile the val datatype
                    realline.append(val) #add to realline index
                    realindex += 1
                    continue
                self.cols[realindex] + self.compiler(val)
                realline.append(val)
                realindex += 1
            else: #otherwise add the skipped row too
                realindex += 1
            index += 1
        # print("ADDING ROW:" , line)
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

    def clusterlabels(self, f): #for every leaf, given a leaf table
        clabel = None #majority of the x values' class
        xlabel = None
        match = 0

        for i, col in enumerate(self.cols): #gets the mode of the class col for the leaf this is the cluster label
            if i in self.y:
                if i in self.skip:
                    continue
                if i in self.nums:
                    clabel = col.median
                else:
                    clabel = col.mode

        for v in self.rows:
            if v not in self.skip:
                xlabel = str(v[len(v)-1])
                if xlabel == clabel:
                    match += 1

        print('length of y:', len(self.rows))
        self.clabels = [clabel for i in range(len(self.rows))]
        print('length of clabels after populating:', len(self.clabels))
        matches = match/(len(self.rows)-1)
        if matches >= 0.8:
            f.write("--------------------------------> Good Cluster Label <--------" +"\n")
        else:
            f.write("Bad Cluster Label" +"\n")

        percent = "{0:.0%}".format(match/(len(self.rows)-1), 2)
        f.write("Cluster Label: " + str(clabel) +"\n")
        f.write("Label Matches: " + str(match) + "/" + str(len(self.rows)-1)+"\n")
        f.write("Label Matches Percentage: " + str(percent) +"\n")



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

def nodes(root): # gets all the nodes
    if root:
        for node in nodes(root.leftNode): yield node
        if root.leaf:  yield root
        for node in nodes(root.rightNode): yield node

def names(root:TreeNode): #gets all the unique ids of the node
  for node in nodes(root):
    print(node.leftTable.cols[1].uid)

def rowSize(t): return len(t.leftTable.rows) #gets the size of the rows

def small2Big(root,how=None): # for all of the leaves from smallest to largest print len of rows & median
  for leaf in sorted(nodes(root), key=how or rowSize):
    t = leaf.leftTable
    print(len(t.rows), [col.mid() for col in t.cols])

    def breadth_first_search(self, f):
     # """In BFS the Node Values at each level of the Tree are traversed before going to next level"""
        count = 0
        to_visit = []
        to_visit.append(self)

        while len(to_visit) != 0:
            current = to_visit.pop(0)
            # print("current node:" , str(current.uid))
            if current.leaf:
                count +=1
                # print("leaf # " , str(count))
                f.write("------------------------------------------------------------------------------------------------------------------------------------------------------" + "\n")
                f.write("------------------------------------------------------------------------------------------------------------------------------------------------------" + "\n")
                # f.write("------------------------------------------------------------------------------------------------------------------------------------------------------" + "\n")
                # f.write("Y Dump Leaf Node/Table: " + str(current.uid) + "\n")
                # current.leftTable.ydump(f)
                # f.write("--------------------------------------------------" + "\n")
                # f.write("X Dump Leaf Node/Table: " + str(current.uid) + "\n")
                # current.leftTable.xdump(f)
                # f.write("--------------------------------------------------" + "\n")
                f.write("Leaf Cluster Labels: " + str(current.uid) + "\n")
                current.leftTable.clusterlabels(f)

            if current.leftNode is not None:
                to_visit.append(current.leftNode)

            if current.rightNode is not None:
                to_visit.append(current.rightNode)

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
    print("#",('{0:20s} {1:11s}  {2:4s}  {3:4s} {4:4s} '+ \
           '{5:4s}{6:4s} {7:3s} {8:3s} {9:3s} '+ \
           '{10:3s} {11:3s}{12:3s}{13:10s}').format(
      "db", "rx",
     "n", "a","b","c","d","acc","pd","pf","prec",
      "f","g","class"))
    print('-'*100)

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
      print("#",('{0:20s} {1:10s} {2:4d} {3:4d} {4:4d} '+ \
          '{5:4d} {6:4d} {7:4d} {8:3d} {9:3d} '+ \
         '{10:3d} {11:3d} {12:3d} {13:10s}').format(i.db,
          i.rx,  n(b + d), n(a), n(b),n(c), n(d),
          p(acc), p(pd), p(pf), p(prec), p(f), p(g),x))
      #print x,p(pd),p(prec)


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
    print("DS 1: Diabetes Case:")
    print("---------------------------")

    lines = Table.readfile("diabetes.csv")
    table = Table(1)
    ls = table.linemaker(lines)
    table + ls[0]
    for l in ls[1:]:
        table + l
    print("CSV --> Table done ...")

    print("Shuffling rows ...")
    random.shuffle(table.rows)
    print("Clustering ...")
    root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    small2Big(root) #bfs for the leaves
    # print("Clustering ...")
    # with open("diabetes_BFS.csv", "w") as f:
    #     root.breadth_first_search(f)

    # print("BFS for cluster labels ...")
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

    print("---------------------------")
    print("--- completed")
    print("---------------------------")
    #
    # print("---------------------------")
    # print("DS 2: Adult Census Case:")
    # print("---------------------------")
    #
    # lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/adultscensusincome.csv") #/Users/laurenalvarez/Desktop/mysublime/datasets/adultscensusincome.csv
    # table = Table(2)
    # ls = table.linemaker(lines)
    # table + ls[0]
    # for l in ls[1:]:
    #     table + l
    # print("CSV --> Table done ...")
    # # print("Printing Raw y-vals ...")
    # # with open("adultcensusincome_raw_y.csv", "w") as f:
    # #     table.ydump(f)
    # print("Shuffling rows ...")
    # random.shuffle(table.rows)
    # print("Clustering ...")
    # root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    #
    # print("BFS for cluster labels ...")
    # with open("adultcensusincome_BFS.csv", "w") as f:
    #     root.breadth_first_search(f)
    # print("Evaluation ...")
    # abcd = Abcd(db='randomIn',rx='all')
    # train = table.clabels
    # test  = table.y
    # print("how many cluster labels:", len(table.clabels))
    # print("how many test/y labels:", len(table.y))
    # # random.shuffle(test)
    # for actual, predicted in zip(train,test):
    #     abcd.tell(actual,predicted)
    # abcd.header()
    # # abcd.ask()
    #
    # print("---------------------------")
    # print("--- completed")
    # print("---------------------------")

    # print("---------------------------")
    # print("DS 3: Banking Case:")
    # print("---------------------------")
    #
    # lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/bankmarketing.csv")
    # table = Table(3)
    # ls = table.linemaker(lines)
    # table + ls[0]
    # for l in ls[1:]:
    #     table + l
    #
    # print("Printing Raw y-vals ...")
    # with open("bankmarketing_raw_y.csv", "w") as f:
    #     table.ydump(f)
    #
    # root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    #
    # print("Clustering ...")
    # with open("bankmarketing_BFS.csv", "w") as f:
    #     root.breadth_first_search(f)
    #
    # print("---------------------------")
    # print("--- completed")
    # print("---------------------------")

    # print("---------------------------")
    # print("DS 4: COMPAS Case:") #ERROR NO ROWS
    # print("---------------------------")
    #
    # lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/compas-scores-two-years.csv")
    # table = Table(4)
    # ls = table.linemaker(lines)
    #
    # table + ls[0]
    # print("ls header:", ls[0])
    # print("header length:", len(ls[0]))
    # for l in ls[1:]:
    #     print("adding line:", l, "length", len(l)) #ERROR ADDING THE LINE: inserting row X of size 51 expected =  53
    #     table + l
    #
    # print("first pass table:", table.rows)
    # # print("Printing Raw y-vals ...")
    # # with open("compas_raw_y.csv", "w") as f:
    # #     table.ydump(f)
    #
    # print("Clustering ...")
    # root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    #
    # with open("compas_BFS.csv", "w") as f:
    #     root.breadth_first_search(f)
    #
    # print("---------------------------")
    # print("--- completed")
    # print("---------------------------")

    # print("---------------------------")
    # print("DS 5: Default Credit Case:")
    # print("---------------------------")
    #
    # lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/defaultcredit.csv")
    # table = Table(5)
    # ls = table.linemaker(lines)
    # table + ls[0]
    # for l in ls[1:]:
    #     table + l
    #
    # print("Printing Raw y-vals ...")
    # with open("defaultcredit_raw_y.csv", "w") as f:
    #     table.ydump(f)
    #
    # print("Clustering ...")
    # root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    #
    # with open("defaultcredit_BFS.csv", "w") as f:
    #     root.breadth_first_search(f)
    #
    # print("---------------------------")
    # print("--- completed")
    # print("---------------------------")

    # print("---------------------------")
    # print("DS 6: German Case:")
    # print("---------------------------")
    #
    # lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/GermanCredit.csv")
    # table = Table(6)
    # ls = table.linemaker(lines)
    # table + ls[0]
    # for l in ls[1:]:
    #     table + l
    #
    # print("Printing Raw y-vals ...")
    # with open("GermanCredit_raw_y.csv", "w") as f:
    #     table.ydump(f)
    #
    # print("Clustering ...")
    # root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    #
    # with open("GermanCredit_BFS.csv", "w") as f:
    #     root.breadth_first_search(f)
    #
    # print("---------------------------")
    # print("--- completed")
    # print("---------------------------")
    #
    # print("---------------------------")
    # print("DS 7: Home Credit Case:") # loaded 266113 rows after 2 hours; error on compiling sym/num cols
    # print("---------------------------")
    #
    # lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/homecreditapplication_train.csv")
    # table = Table(7)
    # ls = table.linemaker(lines)
    # table + ls[0]
    # for l in ls[1:]:
    #     table + l
    # print("CSV --> Table done ...")
    # # print("Printing Raw y-vals ...")
    # # with open("homecredit_raw_y.csv", "w") as f:
    # #     table.ydump(f)
    #
    # print("Clustering ...")
    # root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    # print("BFS for cluster labels ...")
    # with open("homecredit_BFS.csv", "w") as f:
    #     root.breadth_first_search(f)
    #
    # print("---------------------------")
    # print("--- completed")
    # print("---------------------------")
    #
    # print("---------------------------")
    # print("DS 8: Cleveland Case:")
    # print("---------------------------")
    #
    # lines = Table.readfile("/Users/laurenalvarez/Desktop/mysublime/datasets/processed.clevelandhearthealth.csv")
    # table = Table(7)
    # ls = table.linemaker(lines)
    # table + ls[0]
    # for l in ls[1:]:
    #     table + l
    #
    # print("Printing Raw y-vals ...")
    # with open("cleveland_raw_y.csv", "w") as f:
    #     table.ydump(f)
    #
    # print("Clustering ...")
    # root = Table.clusters(table.rows, table, int(math.sqrt(len(table.rows))))
    #
    # with open("cleveland_BFS.csv", "w") as f:
    #     root.breadth_first_search(f)
    #
    # print("---------------------------")
    # print("--- completed")
    # print("---------------------------")


if __name__ == '__main__':
    main()
