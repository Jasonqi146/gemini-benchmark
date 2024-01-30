import rich

with open('sotopia_format.txt', 'r') as file:
    format = str(file.read())
format = format.replace('\\n', '\n')
print(format)