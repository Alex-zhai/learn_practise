while True:
    try:
        nums = map(int, input().split())
        nums.sort()
        diff = 1
        res = []
        for i,j in enumerate(nums):
            if j-i != diff:
                res.append(i + diff)
                diff += 1
        a = int(''.join(map(str, res)))
        print(a%7)
    except:
        break