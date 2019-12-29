while True:
    try:
        str = input()
        len1 = len(str)
        if len1 <= 6:
            print(str)
        elif 6 < len1 <= 14:
            print(str[:6], str[6:14])
        else:
            print(str[:6], str[6:14], str[14:])
    except:
        break

