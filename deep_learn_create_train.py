import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import os
from sklearn.model_selection import train_test_split
import pickle


data_path = Path().absolute() / "original_data"
data_path_test=data_path / "test/test"
data_path_train=data_path / "train/train"


def createImageIndexList(number, start=1, end=100, typeofimage="normal"):   
    pulled_im = []
    label_im=[]
    arr = np.random.choice(np.arange(start, end + 1), size=number, replace=False)
    for index in arr:
        image = typeofimage+ "." + str(index) + ".jpg"
        label_im.append(image)
        what=plt.imread(data_path_train / image)
        # pulled_im.append(cv2.resize(what,(size_x, size_y)))
        pulled_im.append(what)
    return pulled_im, label_im

def createImageList(number, start=1, end=100):
    pulled_im = []
    label_im = []
    pulled_norm_im = []
    pulled_fall_im = []

    tmp_label = [] 
    fall_label = []
    norm_label = []

    arr = np.random.choice(np.arange(start, end + 1), size=number, replace=False)
    data_path = Path("D:/source/Falldetector/data")

    #files = [file for file in data_path.iterdir()]

    #print(len(files))

    with open("d:/source/FALLDETECTOR/train/the_truth.txt", "r") as f:
        tmp_label = [line.strip() for line in f.readlines()] 

    for x in tmp_label:
        x1=x.split(':')[-1]
        filename=x.split(':')[0]
        if x1 =='1':
            fall_label.append('fall')
        else: 
            norm_label.append('normal')
                
        datap = data_path/filename
    #for fi in range(len(files)):
        label_im.append(datap)
        #pulled_im.append(files[fi].resolve())
        image=plt.imread(datap)
        image_rezised = cv2.resize(image, (100, 100))
        if x1 == '1':
            pulled_fall_im.append(image_rezised)

        else:
            pulled_norm_im.append(image_rezised)

    return fall_label, norm_label, pulled_fall_im, pulled_norm_im, len(tmp_label)

def save_data(pathfile, the_data):
    f = open(pathfile, 'wb')
    pickle.dump(the_data, f)
    f.close()

fall_label, norm_label, fall_im, norm_im, no = createImageList(5,0,58365)

print(f" antal {no} fall lab {len(fall_label)}, norm lab {len(norm_label)}, fall no {len(fall_im)}, norm no {len(norm_im)}")


#print(f" all {len(all_im)}, lab all {len(label_im)}, fall {len(fall_im)}, norm {len(norm_im)}")
# fall 1487, lab all 57003, fall 1487, norm 56878




# when fall occurs, devide train and test
X_train_fall, X_test_fall, y_train_fall, y_test_fall = train_test_split(
    fall_im, fall_label, test_size=0.2, random_state=42)

# reduce the normal cases i.e 5% of all (ignore the 95% since risk for biased training)
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(
    norm_im, norm_label, test_size=0.95, random_state=42)

# X_train_norm, y_train_norm contains only 5% of the data

# fixing the normal into train and test
X_train3_norm, X_test3_norm, y_train3_norm, y_test3_norm = train_test_split(
    X_train_norm, y_train_norm, test_size=0.2, random_state=42)

# X_train3_norm + X_test2_norm = X_train_norm (y_train3_norm + y_test3_nrom = y_train_norm)

# split X_train så vi får ett validerings set
X_train2_fall, X_val_fall, y_train2_fall, y_val_fall = train_test_split(
    X_train_fall, y_train_fall, test_size=0.2, random_state=42)

X_train2_norm, X_val_norm, y_train2_norm, y_val_norm = train_test_split(
    X_train3_norm, y_train3_norm, test_size=0.2, random_state=42)

# merge norm and fall data, full set
X_train=X_train_fall + X_train_norm
y_train = y_train_fall + y_train_norm

X_test=X_test_fall + X_test3_norm
y_test = y_test_fall + y_test3_norm
# tuning sets
#       X_train2
X_train2=X_train2_fall + X_train2_norm
y_train2=y_train2_fall + y_train2_norm
#       X_val
X_val=X_val_fall + X_val_norm
y_val=y_val_fall + y_val_norm

save_data("normalized-data/train.dat", [X_train, y_train])
save_data("normalized-data/test.dat",[X_test, y_test])
save_data("normalized-data/train2.dat",[X_train2, y_train2]) # train with validatio
save_data("normalized-data/validate.dat",[X_val, y_val]) # validation

print(f"total training set {len(X_train)}, {len(y_train)}")
print(f"total test set {len(X_test)}, {len(y_test)}")
print(f"total training (using val) set {len(X_train2)}, {len(y_train2)}")
print(f"total validation set {len(X_val)}, {len(y_val)}")
