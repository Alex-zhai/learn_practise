while True:
    try:
        n = int(input())
        strs = []
        for i in range(n):
            str = input()
            if(str != "stop"):
                strs.append(str)
            else:
                break
        sort_strs = sorted(strs, key=lambda x: len(x))
        # sort_strs = sorted(strs, key=len)
        for str in sort_strs:
            print(str)
    except:
        break