while True:
    try:
        str = input()
        dict1 = {}
        for i in str:
            if i.isalpha() or i.isdigit() or i.isspace():
                if i in dict1.keys():
                    dict1[i] += 1
                else:
                    dict1[i] = 1
        sortedList = sorted(sorted(dict1.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)
        str1 = ''.join(item[0] for item in sortedList)
        print(str1)
    except:
        break