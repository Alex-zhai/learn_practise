from collections import Counter
c = Counter(list(map(int, input().strip().split())))
print(c.most_common(1)[0][0])
