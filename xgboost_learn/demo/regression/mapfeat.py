#!/usr/bin/python

fr = open('machine.data', 'r')
fo = open('machine.txt', 'w')

cnt = 6
fmap = {}
while 1:
    line = fr.readline()
    line = line.strip()
    if not line:
        break
    arr = line.split(',')
    fo.write(arr[8])
    fo.write('\t')
    for i in range(6):
        fo.write('%d:%s' % (i, arr[i+2]))
        fo.write('\t')
    if arr[0] not in fmap:
        fmap[arr[0]] = cnt
        cnt += 1
    fo.write('%d:1' % fmap[arr[0]])
    fo.write('\n')
fo.close()






# fo = open('machine.txt', 'w')
# cnt = 6
# fmap = {}
# for l in open( 'machine.data' ):
#     arr = l.split(',')
#     #print(arr[8])
#     fo.write(arr[8])
#     for i in range(0,6):
#         fo.write('%d:%s' %(i,arr[i+2]))
#
#     if arr[0] not in fmap:
#         fmap[arr[0]] = cnt
#         cnt += 1
#     #print(cnt)
#     fo.write('%d:1' % fmap[arr[0]] )  # 对应分类特征  特征值：1
#     #print(fmap[arr[0]])
#     fo.write('\n')
#
# #print(fmap)
# fo.close()

# create feature map for machine data
fo = open('featmap.txt', 'w')
# list from machine.names
names = ['vendor','MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP' ];

for i in range(0,6):
    fo.write( '%d\t%s\tint\n' % (i, names[i+1]))


for v, k in sorted( fmap.items(), key = lambda x:x[1] ):
    fo.write( '%d\tvendor=%s\ti\n' % (k, v))
fo.close()
