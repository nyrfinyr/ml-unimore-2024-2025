import numpy as np
import matplotlib.pyplot as plt

from labs.lab1.components import NonParametricNaiveBayesClassifier
from sklearn.metrics import confusion_matrix

def load_mnist_digits():
    x_train = np.load('mnist/x_train.npy')
    y_train = np.load('mnist/y_train.npy')
    x_test = np.load('mnist/x_test.npy')
    y_test = np.load('mnist/y_test.npy')
    label_dict = {i: str(i) for i in range(0, 10)}
    return x_train, y_train, x_test, y_test, label_dict

def load_mnist(threshold=0.5):
    x_train, y_train, x_test, y_test, label_dict = load_mnist_digits()
    x_train = np.float32(x_train) / 255.
    x_train[x_train >= threshold] = 1
    x_train[x_train < threshold] = 0
    x_test = np.float32(x_test) / 255.
    x_test[x_test >= threshold] = 1
    x_test[x_test < threshold] = 0

    return x_train, y_train, x_test, y_test, label_dict

def plot_classes(x_train, y_train):
    num_row, num_col = 1, 10
    len_tr = len(x_train)
    f, subplots = plt.subplots(num_row, num_col, sharex='col', sharey='row')

    for cls in np.unique(y_train):
        idx = np.arange(len_tr)[y_train == cls]
        idx = np.random.choice(idx)
        X_img = x_train[idx]
        subplots[cls].imshow(X_img, cmap='gray',
                             interpolation='nearest', aspect='auto')
        subplots[cls].set_title(f'Digit {cls}', fontweight="bold")
        subplots[cls].grid(visible=False)
        subplots[cls].axis('off')

    f.set_size_inches(22.5, 4.5)
    f.show()

def get_accuracy(predictions, y_test):
    acc = 0
    for sample in zip(predictions, y_test):
        acc += 1 if sample[0] == sample[1] else 0
    return np.divide(acc, len(y_test))

def main():
    x_train, y_train, x_test, y_test, label_dict = load_mnist(threshold=0.5)
    nbc = NonParametricNaiveBayesClassifier()
    nbc.fit(x_train, y_train)
    predictions = nbc.predict(x_test)
    print(f'Accuracy: {get_accuracy(predictions, y_test)}')



if __name__ == '__main__':
    main()