# install required packages
!pip install medmnist

# import required libraries
from medmnist import BloodMNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load BlooMNIST dataset
dataset = BloodMNIST(split='train',download = False,size=64)
