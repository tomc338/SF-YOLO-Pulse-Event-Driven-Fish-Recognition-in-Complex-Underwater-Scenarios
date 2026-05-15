import numpy as np
boxes = np.load("D:\神经脉冲网络\\research原\czy\czy\zj1class0\class0_1.npz")
print(boxes)
print(boxes["pos"])
print(boxes["neg"])
print(boxes["pos"].shape)