import numpy as np
from matplotlib import pyplot as plt
import os

dir_test = "D:\\test"
for filename in os.listdir(dir_test):
    filen = os.path.join(dir_test, filename)
    img = np.genfromtxt(filen, delimiter = ",")
    plt.figure()
    plt.imshow(img)
    plt.show()
