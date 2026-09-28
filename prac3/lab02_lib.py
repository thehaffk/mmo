# Библиотека преподавателя lab02_lib.py, перенесённая на актуальные версии библиотек.
# Логика функций не менялась. Изменения:
#   - модуль sklearn.datasets.samples_generator удалён в scikit-learn 0.24
#   - datagen: добавлен random_state, иначе набор каждый раз новый
#   - EarlyStoppingOnValue: в Keras 3 обход конструктора Callback ломает self.model
#   - create_fit_save_ae: вместо input() архитектура передаётся параметром arch, ноутбук выполняется без ручного ввода
#   - Adam: аргумент lr удалён в TensorFlow 2.11, теперь learning_rate
#   - create_fit_save_ae: к модели прикреплены MSE_stop и число пройденных эпох
# Проверено на Python 3.12, TensorFlow 2.21, Keras 3.15, scikit-learn 1.9.
#import helpers21
import math
from pandas import DataFrame


import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn
from sklearn import preprocessing
from sklearn.metrics import  precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import zero_one_loss
from sklearn import svm
from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
import matplotlib.pylab as plt
#import aes_lib as aes
import tensorflow.keras
import numpy as np
import math
import matplotlib.pylab as plt
from sklearn import svm
import numpy as np
from pandas import DataFrame
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
#from tensorflow.keras.utils import to_categorical
#from sklearn.metrics import zero_one_loss
#from sklearn.metrics import confusion_matrix


visual = True
verbose_show = False

# generate 2d classification dataset
def datagen(x_c, y_c, n_samples, n_features, random_state=None):

    center = [[x_c, y_c]] if n_features == 2 else None
    X, Y = make_blobs(n_samples = n_samples, centers = center, n_features = n_features, cluster_std = 0.1,
                      random_state = random_state)
    if n_features == 2:
        plt.figure(figsize=(12, 8))
        plt.scatter(X[:,0], X[:,1], marker='o', s=7, color = 'b', label = 'Training set')
        plt.legend(loc = 'upper left', fontsize = 12)
        plt.title('Training set')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('out/train_set.png')
        plt.show()
    np.savetxt('data.txt', X)

    return X


def ire(vector1, vector2):
    x = 0
    for i in range(len(vector1)):
        x += (vector2[i] - vector1[i])**2
    # !! Round to .xx
    ire = round(math.sqrt(x), 2)
    return ire

def ire_array(array1, array2):
    ire_list = []
    for index in range(array1.shape[0]):
        ire_list.append(ire(array1[index], array2[index]))
    ire_array = np.array(ire_list)
    return ire_array

class EarlyStoppingOnValue(tensorflow.keras.callbacks.Callback):

    def __init__(self, monitor='loss', baseline=None):
        super().__init__()
        self.baseline = baseline
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        current_value = self.get_monitor_value(logs)
        if current_value < self.baseline:
            self.model.stop_training = True

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value

#создание и обучение модели автокодировщика
def create_fit_save_ae(cl_train, ae_file, irefile, epohs, verbose_show, patience, arch=None):

    size = cl_train.shape[1]
    # arch: список размеров скрытых слоёв, например [3, 1, 3]; None = архитектура по умолчанию
    ans = '1' if arch is not None else '2'
    if ans == '1':
        ae_arch = list(arch)
        ae = tensorflow.keras.models.Sequential()
        ae.add(tensorflow.keras.layers.Dense(size))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        for i in range(len(ae_arch)):
            ae.add(tensorflow.keras.layers.Dense(ae_arch[i]))
            ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(size))
        ae.add(tensorflow.keras.layers.Activation('linear'))
    else:
        ae = tensorflow.keras.models.Sequential()
        ae.add(tensorflow.keras.layers.Dense(size))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(3))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        #ae.add(tensorflow.keras.layers.Dense(4))
        #ae.add(tensorflow.keras.layers.Activation('tanh'))
        #ae.add(tensorflow.keras.layers.Dense(5))
        #ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(2))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(1))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(2))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        #ae.add(tensorflow.keras.layers.Dense(5))
        #ae.add(tensorflow.keras.layers.Activation('tanh'))
        #ae.add(tensorflow.keras.layers.Dense(4))
        #ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(3))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(size))
        ae.add(tensorflow.keras.layers.Activation('linear'))

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    ae.compile(loss='mean_squared_error', optimizer=optimizer)
    error_stop = 0.0001
    epo = epohs

    early_stopping_callback_on_error = EarlyStoppingOnValue(monitor='loss', baseline=error_stop)
    early_stopping_callback_on_improving = tensorflow.keras.callbacks.EarlyStopping(monitor='loss',
                                                                                           min_delta=0.0001, patience = patience,
                                                                                           verbose=1, mode='auto',
                                                                                           baseline=None,
                                                                                           restore_best_weights=False)
    history_callback = tensorflow.keras.callbacks.History()
    verbose = 1 if verbose_show else 0
    history_object = ae.fit(cl_train, cl_train,
                                batch_size=cl_train.shape[0],
                                epochs=epo,
                                callbacks=[early_stopping_callback_on_error, history_callback,
                                early_stopping_callback_on_improving],
                                verbose=verbose)
    ae_trainned = ae
    ae_pred = ae_trainned.predict(cl_train)
    ae_trainned.save(ae_file)

    IRE_array = np.round(ire_array(cl_train, ae_pred), 2)
    IREth = np.amax(IRE_array)
    with open(irefile, 'w') as file:
        file.write(str(IREth))
    print()
    print()

    # MSE_stop: ошибка, на которой обучение остановилось
    ae_trainned.mse_stop = history_object.history['loss'][-1]
    ae_trainned.epochs_done = len(history_object.history['loss'])

    return ae_trainned, IRE_array, IREth

def test(y_pred, Y_test):
    y_pred[y_pred != Y_test] = -100 # find and mark classification error
    n_errors = (y_pred == -100).astype(int).sum()
    return n_errors

def predict_ae(nn, x_test, threshold):
    x_test_predicted = nn.predict(x_test)
    ire = ire_array(x_test, x_test_predicted)
    # Расчет ошибки при нормализации: иначе закоментировать и раскоментировать 81 и 82
    #x_test_norm = norm_array(x_test, 0)
    #x_test_predicted_norm = nn.predict(x_test_norm)
    #x_test_predicted = norm_array(x_test_predicted_norm, 1)
    #ire = ire_array(x_test, x_test_predicted)
    predicted_labels = (ire > threshold).astype(float)
    predicted_labels = predicted_labels.reshape((predicted_labels.shape[0], 1))
    ire = np.transpose(np.array([ire]))
    return predicted_labels, ire

def load_ae(path_to_ae_file):
    return tensorflow.keras.models.load_model(path_to_ae_file)


def square_calc(numb_square, X_train, ae, IRE_th, num, visual):
    # scan
    x_min, x_max = X_train[:, 0].min() - 2, X_train[:, 0].max() + 1
    # print(x_min, x_max)
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    # print(y_min, y_max)
    h_x = (x_max - x_min) / 100
    h_y = (y_max - y_min) / 100
    h_y = h_x
    #print('ШАГ x:', h_x)
    #print('ШАГ y:', h_y)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))
    X_plot = np.c_[xx.ravel(), yy.ravel()]

    #  получение ответов автоэнкодера
    Z, ire = predict_ae(ae, X_plot, IRE_th)
    # print('z')
    # print(Z)

    X_def = np.array([0, 0], ndmin=2)
    for ind, ans in enumerate(Z):
        if ans == 0:
            # print(ans, ' kl= 1')
            # print(ind, len (svm_predicted_scan))
            X_def = np.append(X_def, [X_plot[ind]], axis=0)

    # построение областей покрытия и границ классов
    X_def = np.delete(X_def, 0, axis=0)
    Z = Z.reshape(xx.shape)

    if visual:
        plt.figure(figsize=(12, 6))
        # fig, ax = plt.subplots()
        plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.5)
        plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=7, color='b')
        plt.legend(['C1'])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Autoencoder AE' + str(num) + '. Training set. Class boundary')
        plt.savefig('out/AE' + str(num) + '_train_def.png')
        plt.show()

    h_x = (x_max - x_min) / numb_square
    h_y = (y_max - y_min) / numb_square
    h_x = abs(h_x)
    h_y = abs(h_y)

    col_id = np.zeros(numb_square)
    col_id_ae = np.zeros(numb_square)

    for i in range(numb_square):
        for x in X_train[:, 0]:
            if x_min + i * h_x <= x < x_min + (i + 1) * h_x:
                col_id[i] = 1
        for x in X_def[:, 0]:
            if x_min + i * h_x <= x < x_min + (i + 1) * h_x:
                col_id_ae[i] = 1

    amount = 0
    cart = np.zeros((numb_square, numb_square))
    for_rect = np.array([0, 0], ndmin=2)
    for index, element in enumerate(col_id):
        if element == 1:
            for i in range(numb_square):
                for xy in X_train:
                    if y_min + i * h_y <= xy[1] < y_min + (i + 1) * h_y and x_min + index * h_x <= xy[0] < x_min + (
                            index + 1) * h_x:
                        amount = amount + 1
                        cart[numb_square - i - 1, index] = 1
                        for_rect = np.append(for_rect, np.array([x_min + index * h_x, y_min + i * h_y], ndmin=2),
                                             axis=0)
                        break

    for_rect = np.delete(for_rect, 0, axis=0)
    # print('cart', cart)
    #print('amount: ', amount)

    amount_ae = 0
    cart_ae = np.zeros((numb_square, numb_square))
    for_rect_ae = np.array([0, 0], ndmin=2)
    for index, element in enumerate(col_id_ae):
        if element == 1:
            for i in range(numb_square):
                for xy in X_def:
                    if y_min + i * h_y <= xy[1] < y_min + (i + 1) * h_y and x_min + index * h_x <= xy[0] < x_min + (
                            index + 1) * h_x:
                        amount_ae = amount_ae + 1
                        cart_ae[numb_square - i - 1, index] = 1
                        for_rect_ae = np.append(for_rect_ae, np.array([x_min + index * h_x, y_min + i * h_y], ndmin=2),
                                                axis=0)
                        break

    for_rect_ae = np.delete(for_rect_ae, 0, axis=0)
    # print('cart_ae', cart_ae)
    print('amount: ', amount)
    print('amount_ae: ', amount_ae)

    if visual:
        label0_ae = 'Распознанное AE' + str(num) + ' множество'
        s0_ae = 0.3
        label0 = 'Обучающее множество'
        s0 = 12

        fig = plt.figure(figsize=(16, 7))
        ax_1 = fig.add_subplot(1, 2, 1)
        ax_2 = fig.add_subplot(1, 2, 2)

        ax_1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
        ax_1.set_xticks(np.arange(x_min, x_max, h_x))
        ax_1.set_yticks(np.arange(y_min, y_max, h_y))
        x_lbl = np.round(np.arange(x_min, x_max, h_x), 1).tolist()
        y_lbl = np.round(np.arange(y_min, y_max, h_y), 1).tolist()

        ax_1.set_xticklabels(x_lbl)
        ax_1.set_yticklabels(y_lbl)

        for xy in for_rect:
            rect = patches.Rectangle((xy[0], xy[1]), h_x, h_y, linewidth=1, edgecolor='none', facecolor='royalblue',
                                     alpha=0.3)
            ax_1.add_patch(rect)

        ax_1.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=s0, color='indigo', label=label0)
        ax_1.legend(loc='upper left', fontsize=12)
        ax_1.set_title('Площадь обучающего множества |Xt|', fontsize=14)
        ax_1.set_xlabel('X')
        ax_1.set_ylabel('Y')
        ax_1.set_xlim(x_min, x_max)
        ax_1.set_ylim(y_min, y_max)

        ax_2.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
        ax_2.set_xticks(np.arange(x_min, x_max, h_x))
        ax_2.set_yticks(np.arange(y_min, y_max, h_y))
        ax_2.set_xticklabels(x_lbl)
        ax_2.set_yticklabels(y_lbl)

        for xy in for_rect_ae:
            rect = patches.Rectangle((xy[0], xy[1]), h_x, h_y, linewidth=1, edgecolor='none', facecolor='coral', alpha=0.4)
            ax_2.add_patch(rect)

        ax_2.scatter(X_def[:, 0], X_def[:, 1], marker='o', s=s0, color='b', label=label0_ae)
        ax_2.legend(loc='upper left', fontsize=12)
        ax_2.set_title('Площадь деформированного множества |Xd|', fontsize=14)
        ax_2.set_xlabel('X')
        ax_2.set_ylabel('Y')

        ax_2.set_xlim(x_min, x_max)
        ax_2.set_ylim(y_min, y_max)
        # plt.xlim(x_min - 4*h_x ,x_max + 4*h_x)
        # plt.ylim(y_min - 4*h_y, y_max + 4*h_y)
        plt.savefig('out/XtXd_' + str(num) + '.png')
        plt.show()


    if visual:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(22, 8))
        # n = 1

        for ax in axes.flat:
            # ax.set(title='axes_' + str(n), xticks=[], yticks=[])
            # n += 1
            # ax.scatter(X_ov[:, 0], X_ov[:, 1], marker='o', s=s0, color='b', label=label0)
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
            ax.set_xticks(np.arange(x_min, x_max, h_x))
            ax.set_yticks(np.arange(y_min, y_max, h_y))  # +0.7
            x_lbl = np.round(np.arange(x_min, x_max, h_x), 1).tolist()
            y_lbl = np.round(np.arange(y_min, y_max, h_y), 1).tolist()
            ax.set_xticklabels(x_lbl)
            ax.set_yticklabels(y_lbl)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            # ax.set_axis_label_font_size(fontsize=14)
            for xy in for_rect_ae:
                rect = patches.Rectangle((xy[0], xy[1]), h_x, h_y, linewidth=1, edgecolor='k', facecolor='coral')
                ax.add_patch(rect)
            rect = patches.Rectangle((xy[0], xy[1]), h_x, h_y, linewidth=1, edgecolor='none', facecolor='coral',
                                     label='Площадь множества |Xd|')
            ax.add_patch(rect)
            for xy in for_rect:
                rect = patches.Rectangle((xy[0], xy[1]), h_x, h_y, linewidth=1, edgecolor='k', facecolor='royalblue')
                ax.add_patch(rect)
            rect = patches.Rectangle((xy[0], xy[1]), h_x, h_y, linewidth=1, edgecolor='none', facecolor='royalblue',
                                     label='Площадь множества  |Xt|')
            ax.add_patch(rect)

            # ax.scatter(for_rect[0, 0] + 0.1, for_rect[0, 1] + 0.1, marker='o', s=s0, color='cornflowerblue', label='Объем обучающего множества, Xt')
            # ax.scatter(for_rect_ae[0,0]+ 0.1, for_rect_ae[0,1]+ 0.1, marker='o', s=s0, color='darkorange', label='Объем деформированного множества, Xd')
            ax.legend(loc='upper left', fontsize=16)
            ####ax.set_ylim(-2.5, 2.9)
            # ax.set_ylim(-1.7, 2.4)#ae2

        nn = 0
        for xy_ae in for_rect_ae:
            for xy in for_rect:
                if xy_ae[0] == xy[0] and xy_ae[1] == xy[1]:
                    nn = nn + 1
                    if nn == 1:
                        rect1 = patches.Rectangle((xy_ae[0], xy_ae[1]), h_x, h_y, linewidth=1, edgecolor='k',
                                                  facecolor='none', hatch='/', label='Площадь на пересечении |Xt| и |Xd|')
                        axes[2].add_patch(rect1)
                    else:
                        rect1 = patches.Rectangle((xy_ae[0], xy_ae[1]), h_x, h_y, linewidth=1, edgecolor='k',
                                                  facecolor='none', hatch='/')
                        axes[2].add_patch(rect1)

        # now#rect1 = patches.Rectangle((xy_ae[0]- 2*h_x, xy_ae[1]-2*h_y), h_x, h_y, linewidth=1, edgecolor='k', facecolor='none', hatch='/', label='Площадь на пересечении |Xt| и |Xd|')
        # now#axes[2].add_patch(rect1)
        axes[2].legend(loc='upper left', fontsize=16)
        # print('true')
        flag = 1
        n = 0
        for xy_ae in for_rect_ae:
            flag = 1
            for xy in for_rect:
                if xy_ae[0] == xy[0] and xy_ae[1] == xy[1]:
                    # print(xy_ae[0], '!=', xy[0],' and ', xy_ae[1], '!=', xy[1])
                    flag = 0

            if flag == 1:
                n = n + 1
                if n == 1:
                    rect2 = patches.Rectangle((xy_ae[0], xy_ae[1]), h_x, h_y, linewidth=1, edgecolor='k', facecolor='none',
                                              hatch='/', label='Площадь |Xd| за исключением |Xt| (|Xd\Xt|)')
                    axes[0].add_patch(rect2)
                else:
                    rect2 = patches.Rectangle((xy_ae[0], xy_ae[1]), h_x, h_y, linewidth=1, edgecolor='k', facecolor='none',
                                              hatch='/')
                    axes[0].add_patch(rect2)

        rect1 = patches.Rectangle((for_rect_ae[0, 0], for_rect_ae[0, 1]), h_x, h_y, linewidth=1, edgecolor='k',
                                  facecolor='none', label='Площадь |Xt| за исключением |Xd| (|Xt\Xd|)')
        axes[1].add_patch(rect1)
        # now#rect2 = patches.Rectangle((xy_ae[0], xy_ae[1]), h_x, h_y, linewidth=1, edgecolor='k', facecolor='none', hatch='/', label='Площадь |Xd| за исключением |Xt| (|Xd\Xt|)')
        # now#axes[0].add_patch(rect2)
        axes[0].legend(loc='upper left', fontsize=16)
        axes[1].legend(loc='upper left', fontsize=16)
        axes[0].set_title('Excess. AE'  + str(num), fontsize=20)
        axes[1].set_title('Deficit. AE'  + str(num), fontsize=20)
        axes[2].set_title('Coating. AE'  + str(num), fontsize=20)
        plt.savefig('out/XtXd_' + str(num) + '_metrics.png')
        plt.show()

    square_ov = amount * h_x * h_y
    square_ae = amount_ae * h_x * h_y

    print()
    print('Оценка качества AE' + str(num))
    extra_pre_ae = square_ov / square_ae
    # print('square_ov:',  square_ov)
    # print('square_ae:', square_ae)

    Ex = cart_ae - cart
    Excess = np.sum(Ex == 1) / amount
    print('IDEAL = 0. Excess: ', Excess)
    Def = cart - cart_ae
    Deficit = np.sum(Def == 1) / amount
    print('IDEAL = 0. Deficit: ', Deficit)
    cart[cart > 0] = 5
    Coa = cart - cart_ae
    Coating = np.sum(Coa == 4) / amount
    print('IDEAL = 1. Coating: ', Coating)
    summa = Deficit + Coating
    print('summa: ', summa)
    print('IDEAL = 1. Extrapolation precision (Approx): ', extra_pre_ae)
    print()
    print()

    with open('out/result.txt', 'w') as file:
        file.write(
            '------------Оценка качества AE' + str(num) + ' С ПОМОЩЬЮ НОВЫХ МЕТРИК------------' + '\n' + \
            'Approx = ' + str(extra_pre_ae) + '\n' + \
            'Excess = ' + str(Excess) + '\n' + \
            'Deficit = ' + str(Deficit) + '\n' + \
            'Coating = ' + str(Coating) + '\n')

    return xx, yy, Z

#####2D
def plot_xdef(X_train, xx, yy, Z):

    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.5)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=7, color='b')
    plt.legend(['C1'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot2in1(X_train, xx, yy, Z1, Z2):

    plt.subplot(1, 2, 1)
    plot_xdef(X_train, xx, yy, Z1)
    plt.title('Autoencoder AE1')#. Training set. Class boundary')

    plt.subplot(1, 2, 2)
    plot_xdef(X_train, xx, yy, Z2)
    plt.title('Autoencoder AE2')#. Training set. Class boundary')
    plt.savefig('out/AE1_AE2_train_def.png')
    plt.show()


def anomaly_detection_ae(predicted_labels, ire, ire_th):
    ire = np.round(ire,2)
    ire_th = np.round(ire_th, 2)
    if predicted_labels.sum() == 0:
        print("Аномалий не обнаружено")
    else:
        print()
        print('%-10s%-10s%-10s%-10s' % ('i', 'Labels', 'IRE', 'IREth'))
        for i, pred in enumerate(predicted_labels):
            print('%-10s%-10s%-10s%-10s' % (i, pred, ire[i], ire_th))
        print('Обнаружено ', predicted_labels.sum(), ' аномалий')


def plot2in1_anomaly(X_train, xx, yy, Z1, Z2, anomalies):

    plt.subplot(1, 2, 1)
    plot_xdef(X_train, xx, yy, Z1)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], marker='o', s=12, color='r')
    plt.title('Autoencoder AE1')#. Training set. Class boundary')

    plt.subplot(1, 2, 2)
    plot_xdef(X_train, xx, yy, Z2)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], marker='o', s=12, color='r')
    plt.title('Autoencoder AE2')#. Training set. Class boundary')
    plt.savefig('out/AE1_AE2_train_def_anomalies.png')
    plt.show()

def ire_plot(title, IRE_test, IREth, ae_name):

    x = range(1, len(IRE_test) + 1)
    IREth_array = [IREth for x in x]
    plt.figure(figsize = (16, 8))
    plt.title('IRE for ' + title + ' set. ' + ae_name, fontsize = 24)
    plt.plot(x, IRE_test, linestyle = '-', color = 'r', lw = 2, label = 'IRE')
    plt.plot(x, IREth_array, linestyle = '-', color = 'k', lw = 2, label = 'IREth')
    #plt.xlim(0, len(x))
    ymax = 1.5 * max(np.amax(IRE_test), IREth)
    plt.ylim(0, ymax)
    plt.xlabel('Vector number', fontsize = 20)
    plt.ylabel('IRE', fontsize = 20)
    plt.grid()
    plt.legend(loc = 'upper left', fontsize = 16)
    plt.gcf().savefig('out/IRE_' + title + ae_name + '.png')
    plt.show()

    return