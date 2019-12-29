while True:
    try:
        str = list(input()[::-1])
        result = []
        for i in str:
            if i not in result:
                result.append(i)
        print("".join(result))
    except:
        break

