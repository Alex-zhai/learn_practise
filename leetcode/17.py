while True:
    try:
        str = input()
        words = str.split(' ')
        print(' '.join(words[::-1]))
    except:
        break