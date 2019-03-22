import numpy as np
import glob
import math
import shutil
import os

def shuffle(arr):
    np.random.shuffle(arr)
    return arr

def split(arr, percentage=0.3):
    val = int(math.ceil(percentage * len(arr)))
    test = arr[:val]
    train = arr[val:]
    print(len(test), len(train))
    return train, test

def move(arr, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    for xml in arr:
        shutil.move(xml, dest)
    

def main():
    xmls = glob.glob('./*/*/*.xml')
    xmls = shuffle(xmls)
    train, test = split(xmls)
    train_path = './data/annotations/train'
    test_path = './data/annotations/test'
    move(train, train_path)
    move(test, test_path)
main()
