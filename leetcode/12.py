while True:
    try:
        n = int(input())
        a = []
        for i in range(n):
            a.append(input())
        for word in sorted(a):
            print(word)
    except:
        break