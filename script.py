import numpy as np
import pandas as pd
import math
np.random.seed(1)

full_labels = pd.read_csv('labels.csv')


gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]


val = int(math.ceil(0.8 * len(grouped_list)))


train_index = np.random.choice(len(grouped_list), size=val, replace=False)
test_index = np.setdiff1d(list(range(len(grouped_list))), train_index)
print(len(train_index), len(test_index))

train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

print(len(train), len(test))

train.to_csv('train_labels.csv', index=None)
test.to_csv('test_labels.csv', index=None)
