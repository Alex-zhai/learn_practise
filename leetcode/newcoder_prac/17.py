while True:
    try:
        s = input().split()
        m = int(s[0])
        n = int(s[1])
        if m < n:
            temp = m
            m = n
            n = temp
        a = m
        b = n
        while n!=0 :
            c = m%n
            m = n
            n = c
        print(int(a*b/m))
    except:
        break
