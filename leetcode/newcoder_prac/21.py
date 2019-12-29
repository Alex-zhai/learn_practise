def main(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    res = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > res:
                    res = m[i + 1][j + 1]
    return res

while True:
    try:
        str1 = input().lower()
        str2 = input().lower()
        print(main(str1, str2))

    except:
        break