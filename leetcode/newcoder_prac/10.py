while True:
    try:
        str = input()
        r_str = str[::-1]
        if str == r_str:
            print("Yes!")
        else:
            print("No!")
    except:
        break