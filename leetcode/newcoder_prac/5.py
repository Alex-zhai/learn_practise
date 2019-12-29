import re

del_str = input()
while True:
    try:
        print(re.sub(r"(?i)" + del_str, "", input()).replace(" ", ""))
    except:
        break

