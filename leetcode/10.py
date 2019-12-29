import sys
n = int(input())

a = {}

for i in range(n):
    k, v = map(int, sys.stdin.readline().split())
    if k in a:
        a[k] += v
    else:
        a[k] = v

for i in sorted(a.keys()):
    print(i, a[i])