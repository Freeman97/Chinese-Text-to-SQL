import matplotlib as mpl
import numpy as np
mpl.use('Agg')

import matplotlib.pyplot as plt

if __name__ == '__main__':
    enc_length = []
    with open('enc_length_stat', 'r') as f:
        enc_length = f.readlines()
    enc_length = [int(x) for x in enc_length]
    print(max(enc_length))
    print(min(enc_length))
    plt.hist(enc_length, bins='auto', density=True)
    plt.savefig('enc_length_stat.png')
    # plt.show()