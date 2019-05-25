import util

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = util.load_processed_data(util.PREPROCESSED_DATA) # this is how we can load the data for conv nets
    print(np.shape(data))
    util.multi_slice_subplot(data[1])
    util.multi_slice_subplot(data[12])
    util.multi_slice_subplot(data[40])
    plt.show()

if __name__ == '__main__':
    main()