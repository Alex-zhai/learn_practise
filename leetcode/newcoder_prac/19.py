while True:
    try:
        nums = list(map(int, input().split(".")))
        flag = 0
        for i in range(len(nums)):
            if nums[i] < 0 or nums[i] > 255:
                flag = 1
            else:
                pass
        if flag == 0:
            print("YES")
        else:
            print("NO")
    except:
        break