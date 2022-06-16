import re,sys
class o:
  """Return a class can print itself (hiding "private" keys)
  and which can hold methods."""
  def __init__(i, **d)  : i.__dict__.update(d)
  def __repr__(i) : return "{"+ ', '.join( 
    [f":{k} {v}" for k, v in i.__dict__.items() if  k[0] != "_"])+"}"

the=o(protected=1,
      want=-2,
      got=-1)

def anExample():
  rows=[row for row in csv("datasets/diabetes.csv")]
  print(rows)

def csv(file=None, sep=",", ignore=r'([\n\t\r ]|#.*)'):
  """Helper function: read csv rows, skip blank lines, coerce strings
     to numbers, if needed."""
  if file:
    with open(file) as fp:
      for a in fp:
        yield [atom(x) for x in re.sub(ignore, '', a).split(sep)]
  else:
    for a in sys.stdin:
      yield [atom(x) for x in re.sub(ignore, '', a).split(sep)]

def atom(x):
  "Coerce x to the right kind of string (int, float, or string)"
  try: return int(x)
  except Exception:
    try:              return float(x)
    except Exception: return x

 
for at,x in enumerate(sys.argv):
  if x=="-p": the.protected = atom(sys.argv[at+1])
  if x=="-got": the.got = atom(sys.argv[at+1])
  if x=="-want": the.want = atom(sys.argv[at+1])

    
    

for row in csv():   
    print(row[the.protected], row[the.got], row[the.want], sep=",")
#:__name__ == "__main__" and anExample()