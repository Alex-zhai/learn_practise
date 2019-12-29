#!/usr/bin/python
import sys
import random

# if len(sys.argv) < 2:
#     print('Usage:<filename> <k> [nfold = 5]')
#     exit(0)

random.seed(10)

k = 5
if len(sys.argv) > 3:
    nfold = int(sys.argv[3])
else:
    nfold = 5

fi = open('machine.txt', 'r' )
ftr = open('machine.txt'+'.train', 'w' )
fte = open('machine.txt'+'.test', 'w' )
for l in fi:
    if random.randint(1, nfold) == k:
        fte.write(l)
    else:
        ftr.write(l)

fi.close()
ftr.close()
fte.close()

