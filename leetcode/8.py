a = []
for i in range(2):
    a.append(input())

for s in a:
    while len(s) >= 8:
        print(s[:8])
        s = s[8:]

    if 0 < len(s) < 8:
        print(s + '0'*(8-len(s)))

