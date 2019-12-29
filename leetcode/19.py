while True:
    try:
        str = ''.join(input())
        chmap = {}
        for ch in str:
            if ch.isalpha():
                if ch in chmap:
                    chmap[ch] += 1
                else:
                    chmap[ch] = 1
                if chmap[ch] == 3:
                    print(ch)
                    break
    except:
        break