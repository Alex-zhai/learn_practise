while True:
    try:
        n = int(input())
        nums = list(map(int, input().split()))
        print(max(nums), min(nums))
    except:
        break