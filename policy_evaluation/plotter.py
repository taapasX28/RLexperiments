
import matplotlib.pyplot as plt
import numpy as np

loss = np.load('lossarr.npy')
plt.plot(loss)
plt.ylabel('Loss')
plt.xlabel('Episodes')
plt.savefig('loss_dia.png')