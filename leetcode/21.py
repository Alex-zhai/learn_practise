str1 = input()
str2 = input()
res = ''
for ch in str2:
    if ch not in str1:
        res += ch
print(res)