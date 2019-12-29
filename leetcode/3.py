while True:
    try:
        str = input()
        temp = list(str)
        filterStr = filter(lambda x:x.isalpha(), list(str))
        filterStr.sort(key=str.upper)
        j=0
        for i in range(len(temp)):
            if temp[i].isalpha():
                temp[i] = filterStr[j]
                j = j + 1
        print(''.join(temp))
    except:
        break