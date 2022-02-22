from collections import defaultdict
import math
import re
import random
import statistics
from itertools import count

# ------------------------------------------------------------------------------
# Column Class
# ------------------------------------------------------------------------------
class Col:
    def __init__(self, name): #The __init__ method lets the class initialize the object's attributes and serves no other purpose
        self.name = name # self is an instance of the class with all of the attributes & methods

    def __add__(self,v):
        return v

    def diversity(self):
        return 0

    def mid(self):
        return 0

    def dist(self, x, y):
        return 0

# ------------------------------------------------------------------------------
# Symbolic Column Class
# ------------------------------------------------------------------------------
class Sym(Col):
    def __init__(self,name,uid,data=None): #will override Col inheritance (could use super() to inherit)
        Col.__init__(self,name) #invoking the __init__ of the parent class; If you forget to invoke the __init__() of the parent class then its instance variables would not be available to the child class.
        self.n = 0
        self.most = 0
        self.mode = ""
        self.uid = uid #unique id --> it allows for permanence and recalling necessary subtables
        self.count = defaultdict(int) #will never throw a key error bc it will guve default value as missing key
        if data != None: #initializes the empty col with val
            for val in data:
                self + val #calls __add__

    #def __str__(self):
        #print to a sym; overrides print(); could replace dump() TBD


    def __add__(self, v):
        self.n += 1 # add value to the count
        self.count[v] += 1 # add value to the dictionary with count +1
        tmp = self.count[v]
        if tmp > self.most: #check which is the most seen; if it's the most then assign and update mode
            self.most = tmp
            self.mode = v
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
        if (x == "?" or x == "") or (y == "?" or y == ""):
            return 1
        return 0 if x == y else 1

# ------------------------------------------------------------------------------
# Numeric Column Class
# ------------------------------------------------------------------------------
class Num(Col):
    def __init__(self, name, uid, data=None):
        Col.__init__(self, name)
        self.n = 0
        self.mu = 0 #
        self.m2 = 0 # for moving std dev
        self.sd = 0
        self.lo = float('inf')
        self.hi = -float('inf')
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
            self. m2 += d * (v - self.mu) # calculate momentumn of the series in realtionship to the distance
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
        return statistics.median(self.vals)


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
        self.xnums = []
        self.xsyms = []
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
            if val[0] == "?" or val[0] == ":": #check the first item is missing/: then skip it; add to skip list
                self.skip.append(index + 1) # index begins with 1
            if val[0].isupper() or "-" in val or "+" in val:
                self.nums.append(index+1) # add to num
                self.cols.append(Num(''.join(c for c in val if not c in ['?',':']), index+1)) # take all the items in val as long as it's not ?/: ;join()takes all items in an iterable and joins them as a string
            else:
                self.syms.append(index+1)
                self.cols.append(Sym(''.join(c for c in val if not c in ['?',":"]), index+1))

            if "!" in val or "-" in val or "+" in val: #for any goal, or klass add to y
                self.y.append(index+1)
                self.goals.append(index+1)
                if "-" in val:
                    self.w[index+1] = -1
                if "+" in val:
                    self.w[index+1] = 1
                if "!" in val:
                    self.klass.append(index+1)
            if "-" not in val and "+" not in val and "!" not in val: #catch the rest and add to x
                self.x.append(index+1)
                if val[0].isupper(): #check is num
                    self.xnums.append(index + 1)
                else: #else add to sym
                    self.xsyms.append(index+1)
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
            if index+1 not in self.skip: #check if it needs to be skipped
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
    #print("num_rows:", num_rows)
    assert 101 == num_rows, "counting rows"


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    test_num()
    test_sym()
    test_rows()
    print("---------------------------")
    print("All 3 unit tests passed")
    print("---------------------------")
    ####### FULL TEST #########
    lines = Table.readfile("test.csv")
    table = Table(0)
    ls = table.linemaker(lines)
    for line in ls:
        table + line
    print("---------------------------")
    print("Testing Code with small fake data: Test.csv ")
    print("Test Table Cols:", table.cols)
    print("Test Table Num Cols Vals:", table.cols[0].vals)
    print("Test Table Sym Cols Dictionary:", table.cols[1].count)
    print("Test Table Num Cols Mid:", table.cols[0].median)
    print("Test Table Sym Cols Mid:", table.cols[1].mode)
    print("Test Table Rows:", len(table.rows))
    print("Test Table Skips:", len(table.skip))
    print("Test Table Goals:", len(table.goals))
    print("Test Table Klass:", len(table.klass))
    print("Test Table Header:", table.header)
    print("Test Table Nums:", len(table.nums))
    print("Test Table Syms:", len(table.syms))
    print("Test Table xNums:", table.xnums)
    print("Test Table xSyms:", table.xsyms[0]) #why is this 2???
    ##########################
    print("---------------------------")
    print("---------------------------")
    print("Final run completed")
    print("---------------------------")


if __name__ == '__main__':
    main()
