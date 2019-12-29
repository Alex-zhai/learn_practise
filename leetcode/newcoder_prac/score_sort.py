try:
    while 1:
        num = int(input())
        flag = int(input())
        dict1 = {}
        for i in range(num):
            line = input()
            name = line.split()[0]
            score = line.split()[1]
            dict1[name] = score
        if flag:
            sorted_list = sorted(dict1.items(), key=lambda x: int(x[1]), reverse=False)
        else:
            sorted_list = sorted(dict1.items(), key=lambda x: int(x[1]), reverse=True)
        for item in sorted_list:
            print(item[0] + " " + item[1])
except:
    pass