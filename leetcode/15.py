while True:
    try:
        a = input().strip()
        b = input().strip()
        if a == "" or b == "":
            break
        if len(a) > len(b):
            a, b = b, a
        n = 0
        for i in range(len(a)):
            for j in range(i+1, len(a)):
                sub = a[i:j]
                if sub in b and j-i > n:
                    n = j - i
                    sub1 = sub
        print(''.join(sub1))
    except:
        break