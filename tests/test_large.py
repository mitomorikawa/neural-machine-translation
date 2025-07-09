import csv

with open("../data/eng_fra_large_train.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    i = 0
    for row in reader:
        if i < 10:
            print(row)
        i += 1
   