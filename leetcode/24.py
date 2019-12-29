import math
while True:
    try:
        num, n = map(int, input().split())
        sum1 = num
        for i in range(n-1):
            sum1 += math.sqrt(num)
            num = math.sqrt(num)
        print('{:.2f}'.format(sum1))
        # print('%.2f' %(sum1))
    except:
        break