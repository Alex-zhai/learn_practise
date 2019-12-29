while True:
    try:
        str = input()
        chars = list(str)
        i = 0
        while i < len(chars):
            if chars[i].isdigit():
                tmp = 1
                try:
                    while chars[i+tmp].isdigit():
                        tmp += 1
                except:
                    pass
                chars.insert(tmp+i, '*')
                chars.insert(i,'*')
                i += tmp + 2
            else:
                i+=1
        print(''.join(chars))
    except:
        break
