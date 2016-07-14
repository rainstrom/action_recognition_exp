import sys
with open(sys.argv[1]) as f:
    lines = f.readlines()
    for line in lines:
        splits = line.split(' ')
        splits = [s.strip() for s in splits]
        print splits[0]+".avi", splits[1], splits[2]
