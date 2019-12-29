while True:
    try:
        list1 = list(input().split(" "))
        print(" ".join(list1[::-1]))

    except:
        break