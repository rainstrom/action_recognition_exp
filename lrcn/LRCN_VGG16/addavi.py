import sys


with open(sys.argv[1]) as f:
  lines = f.readlines()
  for line in lines:
    splits = [split.strip() for split in line.split(' ')]
    assert len(splits) == 2
    # print splits
    print splits[0]+".avi", splits[1]
