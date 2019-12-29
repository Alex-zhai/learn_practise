while True:
    try:
        n = int(input())
        nums = list(map(int, input().split()))
        flag = int(input())
        print(nums)
        if flag == 0:
            nums.sort()
            nums = list(map(str, nums))
            print(" ".join(nums))

        else:
            nums.sort(reverse=True)
            nums = list(map(str, nums))
            print(" ".join(nums))
    except:
        break