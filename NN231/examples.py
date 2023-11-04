import random

OUTPUT = "examples.txt"
EXAMPLE_COUNT = 30

examples = []
while (len(examples) != EXAMPLE_COUNT):
    a = round(random.random(), 4)
    b = round(random.random(), 4)
    if a + b <= 1.0:
        examples.append((a, b, round(a + b, 6)))

with open(OUTPUT, 'w') as file:
    file.write(f"{EXAMPLE_COUNT}\n")
    for item in examples:
        file.write(f"{item[0]} {item[1]} {item[2]}\n")
