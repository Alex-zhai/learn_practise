while True:
    try:
        n = int(input())
        for i in range(n):
            strs = input().split()
            r_strs = [str[::-1] for str in strs]
            r_sum = int(r_strs[0]) + int(r_strs[1])
            sum1 = int(str(int(strs[0]) + int(strs[1]))[::-1])
            if r_sum == sum1:
                print(int(strs[0]) + int(strs[1]))
            else:
                print("NO")
    except:
        break