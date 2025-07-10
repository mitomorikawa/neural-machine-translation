import csv
from tqdm import tqdm

with open("../data/eng_fra_large_train.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    i = 0
    for row in tqdm(reader):
        if len(row)!=2:
            print(row, i)
            
        

   