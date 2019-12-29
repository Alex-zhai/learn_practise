while True:
    try:
      num = int(input())
      while num != 1:
          i = 2
          while True:
              if num % i == 0:
                  print(i, end=" ")
                  num /= i
                  break
              i = i + 1
    except:
        break