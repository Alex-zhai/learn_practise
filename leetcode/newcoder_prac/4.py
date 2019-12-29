while True:
    try:
        n = input()
        nums = map(int, input().split())
        nums_sort = sorted(set(nums))
        result = list(map(lambda x:nums_sort.index(x) + 1, nums))
        print(result)
        print("dadas")
        print(" ".join(result))
    except:
        break
