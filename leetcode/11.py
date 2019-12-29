while True:
    try:
        str = input()
        strList = list(str)
        a = {}
        for ch in str:
            if ch in a:
                a[ch] += 1
            else:
                a[ch] = 1
        minCount = min(a.values())
        for i in a.keys():
            if a[i] == minCount:
                strList.remove(i)
        print(''.join(strList))
    except:
        break