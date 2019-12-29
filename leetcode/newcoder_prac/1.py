while True:
    try:
        str = input()
        code = []
        for ch in str:
            if ch.isupper():
                if ch == 'Z':
                    code.append('a')
                else:
                    code.append(chr(ord(ch.lower()) + 1))
            elif ch.islower():
                if ch in ['a', 'b', 'c']:
                    code.append('2')
                elif ch in ['d', 'e', 'f']:
                    code.append('3')
                elif ch in ['g', 'h', 'i']:
                    code.append('4')
                elif ch in ['j', 'k', 'l']:
                    code.append('5')
                elif ch in ['m', 'n', 'o']:
                    code.append('6')
                elif ch in ['p', 'q', 'r', 's']:
                    code.append('7')
                elif ch in ['t', 'u', 'v']:
                    code.append('8')
                else:
                    code.append('9')
            else:
                code.append(ch)
        print("".join(code))
    except:
        break