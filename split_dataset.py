import json
import random

input_file = 'beit3_final_train.jsonl'
train_file = 'beit3_train.jsonl'
val_file = 'beit3_val.jsonl'
test_file = 'beit3_test.jsonl'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Mischia i dati per avere una distribuzione equa
random.seed(42)
random.shuffle(lines)

total = len(lines)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

with open(train_file, 'w', encoding='utf-8') as f:
    f.writelines(lines[:train_end])
with open(val_file, 'w', encoding='utf-8') as f:
    f.writelines(lines[train_end:val_end])
with open(test_file, 'w', encoding='utf-8') as f:
    f.writelines(lines[val_end:])

print(f"Divisione completata:\nTrain: {train_end}\nVal: {val_end-train_end}\nTest: {total-val_end}")

