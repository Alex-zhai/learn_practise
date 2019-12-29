# def getAllSub(str, k):
#     len1 = len(str)
#     res = []
#     for i in range(0, len1-k+1):
#         for j in range(i+k, len1+1):
#             res.append(str[i:j])
#     return res
#
# def getGini(str):
#     res = 0
#     for ch in str:
#         if ch == 'G' or ch == 'C':
#             res += 1
#     return res / len(str)
#
# while True:
#     try:
#         str = input()
#         k = int(input())
#         strList = getAllSub(str, k)
#         maxLen = 0
#         resStr = ''
#         for str in strList:
#             if getGini(str) > maxLen:
#                 maxLen = getGini(str)
#                 print(maxLen)
#                 print(str)
#
#     except:
#         break
while True:
    try:
        str = input()
        n = int(input())
        maxStr, maxLen = str[:n], str[:n].count('C') + str[:n].count('G')
        for i in range(0, len(str)-n):
            if str[i:i+n].count('C') + str[i:i+n].count('G') > maxLen:
                maxLen = str[i:i+n].count('C') + str[i:i+n].count('G')
                maxStr = str[i:i+n]
        print(maxStr)
    except:
        break