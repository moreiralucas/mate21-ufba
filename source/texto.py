import numpy as np
import cv2
import sys
import os

# def get_text(path):
#     f = open(path, 'r')

def get_text(path):
    with open(path, 'r') as f:
        contents = f.read()
        return contents

t1 = get_text("29d18h2m.txt")
t1 = t1.splitlines()
t1.sort()
# print(t1)

path = "../data_part1/test/"

t2 = get_text("/tmp/result.txt")
t2 = t2.splitlines()
t2.sort()
print(len(t1) == len(t2))
cont = 0
images = sorted(os.listdir(path))
# print("images: ")
# print(images)
for i in range(len(t1)):
    if t1[i] != t2[i]:
        print("{} <--> {}".format(t1[i], t2[i]))
        cont += 1
        lucas = t1[i]
        maups = t2[i]

        lab1 = "lucas: " + lucas[len(t1[i]) - 1]
        lab2 = "maups: " +maups[len(t2[i]) - 1]
        
        img = cv2.imread(path+'/'+images[i], cv2.IMREAD_GRAYSCALE)

        label = lab1 + " --- " + lab2
        cv2.imshow(label, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # break

print("cont: {}, total: {}".format(cont, len(t1)))