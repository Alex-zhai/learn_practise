# import pandas as pd
# while True:
#     try:
#         n = int(input())
#         keys = []
#         values = []
#         for i in range(n):
#             nums = input().split(" ")
#             keys.append(int(nums[0]))
#             values.append(int(nums[1]))
#         d = {'key': keys, 'value': values}
#         df = pd.DataFrame(data=d)
#         df1 = df['value'].groupby(df['key']).sum()
#         for i in range(df1.shape[0]):
#             print(df1.index[i], df1.values[i])
#     except:
#         break

while True:
    try:
        n = int(input())
        map1 = {}
        for i in range(n):
            key, value = [int(i) for i in input().split(" ")]
            if key in map1:
                map1[key] += value
            else:
                map1[key] = value
        for key in sorted(map1.keys()):
            print(key, map1[key])
    except:
        break