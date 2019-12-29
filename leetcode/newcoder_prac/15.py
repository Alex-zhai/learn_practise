while True:
    try:
        str = input()
        str1 = []
        count = 0
        for ch in str:
            if 0 <= ord(ch) < 128:
                if ch not in str1:
                    str1.append(ch)
                    count += 1
        print(count)
    except:
        break