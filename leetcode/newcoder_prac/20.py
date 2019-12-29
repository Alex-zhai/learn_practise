# -*- coding: UTF-8 -*-

# def is_symmetry(str):
#     for i in range(len(str)):
#         if str[i] != str[len(str)-i-1]:
#             return False
#     return True
#
# while True:
#     try:
#         str = input()
#         max_len = 1
#         for i in range(len(str)):
#             for j in range(i+1, len(str)+1):
#                 if is_symmetry(str[i:j]):
#                     if j-i > max_len:
#                         max_len = j-i
#         print(max_len)
#     except:
#         break

while True:
    try:
        str = input()
        strLen = len(str)
        p = [[False for i in range(strLen)] for j in range(strLen)]
        maxLen = 0
        for i in range(strLen):
            p[i][i] = True
            if i < strLen - 1 and str[i] == str[i+1]:
                p[i][i+1] = True
                maxLen = 2

        for i in range(3, strLen):  # 子串长度
            for j in range(0, strLen-i):  # 子串起始位置
                end = j + i - 1
                if p[j+1][end-1] and str[j] == str[end]:
                    p[j][end] = True
                    maxLen = i
        print(maxLen)
    except:
        break

