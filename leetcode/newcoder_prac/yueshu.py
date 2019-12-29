while True:
    try:
        nums = list(map(int, input().split()))
        print(sum(filter(lambda x: x < nums[0], nums[1:])))
    except:
        break