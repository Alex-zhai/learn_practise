from functools import cmp_to_key
num = int(input())
nums = input().split(" ")

def cmp(a, b):
    ab = int(a + b)
    ba = int(b + a)
    if ab > ba:
        return 1
    else:
        return -1
nums.sort(key=cmp_to_key(cmp), reverse=True)
print(int(''.join(nums)))