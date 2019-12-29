# from itertools import combinations
# from collections import Counter
# while True:
#     try:
#         str = input()
#         substrs = []
#         for i in range(1, len(str) + 1):
#             substrs.extend(list(combinations(str, i)))
#         substrs = [''.join(i) for i in substrs]
#         print(substrs)
#         count_dict = Counter(substrs)
#         for item in count_dict.items():
#             if item[1] > 2:
#                 print(item[0], item[1])
#     except:
#         break

while True:
    try:
        str = input()
        substrs = {}
        s_len = len(str)
        for i in range(s_len):
            for j in range(i+1, s_len + 1):
                substr = str[i:j]
                if substr in substrs:
                    substrs[substr] += 1
                else:
                    substrs[substr] = 1
        result = sorted(filter(lambda x: x[1] > 1, substrs.items()), key=lambda x: x[0])
        for u, v in result:
            print(u, v)
    except:
        break