with open("data.xlsx", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        print(line)