import sys
n = int(input())
nums = map(int, input().split())

sum = 0
for i in range(n):
    for j in range(i+1, n):
        sum