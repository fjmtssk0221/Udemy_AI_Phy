
#Sequentialはニューラルネットワークのモデルを定義
from keras.models import Sequential
#Conv,MaxPoolは畳み込み等の関数元
from keras.layers import Convolution2D, MaxPool2D
from keras.layers import Activation,Dropout, Flatten,Dense
#numpy用の関数群を定義
from keras.utils import np_utils
import numpy as np

#＜クラス準備＞
classes = ["monkey","boar","crow"]
#クラスのサイズ定義
num_classes = len(classes)
image_size = 50
