
#Sequentialはニューラルネットワークのモデルを定義
from keras.models import Sequential
#Conv,MaxPoolは畳み込み等の関数元
from keras.layers import Conv2D, MaxPool2D
#Kerasの便利関数達
from keras.layers import Activation,Dropout, Flatten,Dense
#numpy用の関数群を定義
from keras.utils import np_utils
import numpy as np

#＜クラス準備＞
classes = ["monkey","boar","crow"]
#クラスのサイズ定義
num_classes = len(classes)
image_size = 50

#mainの関数を定義する
def main():
    X_train,X_test,Y_train,Y_test = np.load("./animap.npy")
    #２５６で正規化する
    X_train = X_train.astype("float")/256 #floatは浮動小数点数
    X_test = X_test.astype("float")/256

    #one-hot-vector：正解は１ほかは０となります
    #[0,1,2]を[1,0,0][0,1,0][0,0,1]となるように
    Y_train = np_utils.to_categorical(Y_train,num_classes)
    Y_test = np_utils.to_categorical(Y_test,num_classes)

    #model関数を
    model = model_train(X_train,Y_train)
    model_eval(model,X_test,Y_test)

#Kerasの
def model_train():
    model = Sequential()
    #padding　畳み込み
    #shape[1:]=>x_trainの形を確認する関数
    #x_trainは[450(個数),50(X),50(Y),3(RGB)]で1項目を除いての[50,50,3]を取り出す
    model.add(Conv2(32,(3,3),padding='same',input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))#非負にする（－は０にする）
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(Maxpooling2D(pool_size=(2,2)))#2×2の中で最も数字の大きいものを取り出す
    model.add(Dropout(0.25))#25％捨てる

    model.add(Conv2D(64,(3,3),padding ='same')
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(Maxpooling2D(pool_size=(2,2)))#2×2の中で最も数字の大きいものを取り出す
    model.add(Dropout(0.25))#25％捨てる

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))#50％捨てる
    model.add(Dense(3))
    model.add(Activation('softmax'))
