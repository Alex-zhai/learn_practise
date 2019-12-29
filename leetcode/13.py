while True:
    try:
        str = input()
        numStr = ''
        for i in range(len(str)):
            if str[i].isdigit():
                numStr += str[i]
            else:
                numStr += '#'
        splitStr = numStr.split('#')
        maxLen = 0
        for i in splitStr:
            if len(i) > maxLen:
                maxLen = len(i)
        res = ''
        for i in splitStr:
            if len(i) == maxLen:
                res += i
        print(res + ',' + str(maxLen))
    except:
        break
