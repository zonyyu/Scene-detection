import numpy as np
from PIL import Image
import glob


def load_data(filename):
    train_f = filename + "/" + "seg_train"
    cv_f = filename + "/" + "seg_test"
    test_f = filename + "/" + "seg_pred"

    classes = ["buildings", "forest", "glacier",
             "mountain", "sea", "street"]
    side = 64

    X_train = np.empty(0)
    Y_train = np.empty(0)
    X_test = np.empty(0)
    Y_test = np.empty(0)

    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []

    print("Loading from train set...")
    for i in range(len(classes)):
        train_subfolder = train_f + "/" + classes[i]
        elem = glob.glob(train_subfolder + "/*.jpg")
        label = np.zeros((1, len(classes)))
        label[0][i] = 1

        for j in range(len(elem)):
            im = Image.open(elem[j])
            im = im.resize((side, side))
            X_train_list.append(np.array(im).reshape(1, side*side*3))
            Y_train_list.append(label)

    print("Loading from test set...")
    for i in range(len(classes)):
        cv_subfolder = cv_f + "/" + classes[i]
        elem = glob.glob(cv_subfolder + "/*.jpg")
        label = np.zeros((1, len(classes)))
        label[0][i] = 1

        for j in range(len(elem)):
            im = Image.open(elem[j])
            im = im.resize((side, side))
            X_test_list.append(np.array(im).reshape(1, side*side*3))
            Y_test_list.append(label)
    

    print("Stacking...")
    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)
    print("Shuffling...")
    np.random.seed(42)
    np.random.shuffle(X_train)
    np.random.seed(42)
    np.random.shuffle(Y_train)
    np.random.seed(24)
    np.random.shuffle(X_test)
    np.random.seed(24)
    np.random.shuffle(Y_test)
    X_train = X_train.T/255
    Y_train = Y_train.T/255
    X_test = X_test.T/255
    Y_test = Y_test.T/255
    print("Done")

    return (X_train, Y_train, X_test, Y_test)


def sigmoid(x):
    return 1/(1 + np.exp(-x))




