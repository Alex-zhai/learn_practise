import re
while True:
    try:
        str = input().strip()
        patten = re.compile('[0-9]*')
        s = patten.findall(str)
        maxLen = 0
        for i in s:
            if len(i) > maxLen:
                maxLen = len(i)
        print(maxLen)
        res = ''
        for i in s:
            if len(i) == maxLen:
                res += i
        print(res + ',' + str(maxLen))
    except:
        break
