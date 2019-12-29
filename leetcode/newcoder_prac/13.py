while True:
    try:
        num = input()
        temp = num.split(".")
        if int(temp[1][0]) >= 5:
            print(int(temp[0]) + 1)
        else:
            print(int(temp[0]))
    except:
        break

