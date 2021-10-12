# -*- coding: utf-8 -*-
"""ensemble_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12cnIi8afGRrRdfPorgvsAFoulSt4y5yp
"""
import tensorflow as tf
from keras.utils import to_categorical
# from tensorflow import keras
import keras
from keras.layers import Conv1D, Input, Activation, Dropout, MaxPooling1D, Flatten, Dense, Conv2D, Reshape
from keras.layers import concatenate, add, Concatenate
from keras import Model
import os
from keras import backend as K
from deslib.dcs.ola import OLA

K.clear_session()  # if flushes the memory accumulated by DL models


def get_data_split(data):
    x = []
    y = []
    for key, value in data.items():
        target = list(value.keys())[0]
        x.append(value[target])
        y.append(target)
    return x, y


def read_pickle_data(path=None):
    import pickle
    with open(path, 'rb') as fid:
        data = pickle.load(fid)
        fid.close()
        return data


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=1)


from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import load_model

train_path = "D:/Anish_PeerJ/dataset/set_2/avg_distance_based_train.pkl"
test_path = "D:/Anish_PeerJ/dataset/set_2/avg_distance_based_test.pkl"

train_data = read_pickle_data(train_path)
test_data = read_pickle_data(test_path)

train_x, train_y = get_data_split(train_data)
test_x, test_y = get_data_split(test_data)

le = LabelEncoder()
le.fit(train_y)

train_y = le.transform(train_y)
test_y__ = le.transform(test_y)

train_y = to_categorical(train_y)
test_y = to_categorical(test_y__)

print(np.shape(train_x), np.shape(train_y))
print(np.shape(test_x), np.shape(test_y))
# model = mcnn()
# print(model.summary())
import numpy as np

# train_x = np.array(train_x)
train_x = np.array(train_x).reshape((np.shape(train_x)[0], np.shape(train_x)[1], 1))
train_y = np.array(train_y)

# test_x = np.array(test_x)
test_x = np.array(test_x).reshape((np.shape(test_x)[0], np.shape(test_x)[1], 1))
test_y = np.array(test_y)

print(np.shape(train_x), np.shape(train_y))
print(np.shape(test_x), np.shape(test_y))
#
# model_x = load_model("D:/Anish_PeerJ/pre_trained_models/set_1/x_set1_16NepaliNews.h5")
#
# model_y = load_model("D:/Anish_PeerJ/pre_trained_models/set_1/y_set1_16NepaliNews.h5")
#
# model_z = load_model("D:/Anish_PeerJ/pre_trained_models/set_1/z_set1_16NepaliNews.h5")
#
# model_t = load_model("D:/Anish_PeerJ/pre_trained_models/set_1/t_set1_16NepaliNews.h5")
#
# model_m = load_model("D:/Anish_PeerJ/pre_trained_models/set_1/m_set1_16NepaliNews.h5")
#
# model_n = load_model("D:/Anish_PeerJ/pre_trained_models/set_1/n_set1_16NepaliNews.h5")
#
# # model.fit(train_x, train_y, epochs=150, verbose=1, validation_data=(test_x, test_y))
#
# models = [model_x, model_y, model_z, model_t, model_m, model_n]
# # define ensemble model
# stacked_model = define_stacked_model(models)
# stacked_model.summary()
# # fit stacked model on test dataset
# history = fit_stacked_model(stacked_model, train_x, train_y, test_x, test_y)
# # make predictions and evaluate

input_tensor_size = Input(shape=(309, 1))


#
# def SuperLearner():
#     # input_tensor = Input(shape=(150, 150, 3))
#     m1 = load_model("D:/Anish_PeerJ/pre_trained_models/set_2/x_set1_16NepaliNews.h5")
#     m2 = load_model("D:/Anish_PeerJ/pre_trained_models/set_2/y_set1_16NepaliNews.h5")
#     m3 = load_model("D:/Anish_PeerJ/pre_trained_models/set_2/z_set1_16NepaliNews.h5")
#     m4 = load_model("D:/Anish_PeerJ/pre_trained_models/set_2/t_set1_16NepaliNews.h5")
#     m5 = load_model("D:/Anish_PeerJ/pre_trained_models/set_2/m_set1_16NepaliNews.h5")
#     m6 = load_model("D:/Anish_PeerJ/pre_trained_models/set_2/n_set1_16NepaliNews.h5")
#
#     m1.name = "m1"  # provide new name for m1 model as default is model_1
#     m2.name = "m2"  # provide new name for m2 model as default is model_1
#     m3.name = "m3"  # provide new name for m3 model as default is model_1
#     m4.name = "m4"  # provide new name for m4 model as default is model_1
#     m5.name = "m5"  # provide new name for m3 model as default is model_1
#     m6.name = "m6"  # provide new name for m4 model as default is model_1
#
#     for layer in m1.layers:
#         layer.name = layer.name + 'first'
#         layer.trainable = True
#
#     for layer in m2.layers:
#         layer.name = layer.name + 'second'
#         layer.trainable =True
#
#     for layer in m3.layers:
#         layer.name = layer.name + 'third'
#         layer.trainable = True
#
#     for layer in m4.layers:
#         layer.name = layer.name + 'fourth'
#         layer.trainable =True
#
#     for layer in m5.layers:
#         layer.name = layer.name + 'third'
#         layer.trainable =True
#
#     for layer in m6.layers:
#         layer.name = layer.name + 'fourth'
#         layer.trainable =True
#
#     m_1 = m1(input_tensor_size)  # it avoids graph disconnect problem
#     m_2 = m2(input_tensor_size)  # it avoids graph disconnect problem
#     m_3 = m3(input_tensor_size)  # it avoids graph disconnect problem
#     m_4 = m4(input_tensor_size)  # it avoids graph disconnect problem
#     m_5 = m5(input_tensor_size)  # it avoids graph disconnect problem
#     m_6 = m6(input_tensor_size)  # it avoids graph disconnect problem
#
#     # m_1=m1.output
#     # m_2=m2.output
#     # m_3=m3.output
#     # m_4=m4.output
#     # m_5=m5.output
#     # m_6=m6.output
#
#     concat = keras.layers.concatenate([m_1, m_2, m_3, m_4, m_5, m_6])
#     print(concat.shape)
#     reshaped_tensor = Reshape(target_shape=(6, 16, 1))(concat)
#     print(reshaped_tensor.shape)
#     layer = Conv2D(256, kernel_size=1, activation='relu', name='conv_new')(reshaped_tensor)
#     layer = Flatten(name='flatten_new')(layer)
#     layer = Dropout(0.5, name='dropout_new')(layer)
#     pred = Dense(16, activation='softmax', name='classification_layer')(layer)
#     print('Prediction size:\n')
#     print(pred)
#     model = Model(inputs=input_tensor_size, outputs=pred, name='new_model')
#     return model


def ensemble():
    models = []
    pre_trained_models = os.listdir('D:/Anish_PeerJ/pre_trained_models/set_2/models/')
    for i in range(6):
        modelTemp = load_model('D:\Anish_PeerJ\pre_trained_models\set_2\models/' + pre_trained_models[i])  # load model
        modelTemp.name = str(i)  # change name to be unique
        models.append(modelTemp)

    # collect outputs of models in a list
    yModels = [model(input_tensor_size) for model in models]
    # averaging outputs
    yAvg = keras.layers.average(yModels)
    modelEns = Model(inputs=input_tensor_size, outputs=yAvg, name='ensemble')

    return modelEns


model_input = Input(shape=(309, 1))  # c*h*w
model = ensemble()
# model=ensemble_deslib()
# model.summary()
#
#
model.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
              optimizer=keras.optimizers.RMSprop(lr=0.00001),
              #  optimizers=optimizers.Adam(lr_schedule),
              metrics=['acc'])

# model.fit(train_x,train_y)


history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=128, epochs=10)

print("Evaluate on test data")
results = model.evaluate(test_x, test_y)
print("test loss, test acc:", results)
#
from matplotlib import pyplot

# learning curves of model accuracy
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
# pyplot.savefig("/content/drive/MyDrive/sets/set_4/o_set_4_train_test_plot.png")
pyplot.show()