while True:
    try:
        n = input()
        a = [int(i) for i in input().split()]
        f = input()
        a.sort()
        b = [str(i) for i in a]
        print (' '.join(b) if f == '0' else ' '.join(b[::-1]))
    except:
        break