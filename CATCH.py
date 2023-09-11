import tensorflow as tf
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras import layers
import time
df = pd.read_csv('./Pre.csv')
df.pop('W_3_24')
df.pop('W_3_25')
df.pop('W_3_26')
df.pop('W_3_27')
df.pop('W_3_28')
df.pop('W_3_29')
df.pop('W_3_30')
df.pop('W_3_31')
df.pop('W_3_32')
df.pop('W_3_33')
df.pop('W_3_34')
df.pop('W_3_35')
df.pop('W_3_36')
df.pop('W_3_37')
df.pop('W_3_38')
df.pop('W_3_39')
df.pop('Unnamed: 0')


cor = df.corr()

for index, row in cor.iterrows():
    if abs(row['yield'])<0.1:
        df.pop(index)

df_mean = df.mean()
df_std = df.std()


df=(df-df_mean)/df_std
temp1 = df.pop('yield')
cor = cor.pop('yield')
df.insert(df.shape[1],'yield',temp1)
seq = pd.DataFrame(df).to_numpy()
#(310*30)*126



def split_seq(Seq,n):
    X = list()
    y = list()
    X1 = list()
    y1 = list()
    X2 = list()
    y2 = list()
    X3 = list()
    y3 = list()
    X4 = list()
    y4 = list()
    cou=-1
    for i in range(len(Seq)):
        
        if (i+n-1)%30==0:
            cou=n-1
        if cou > 0:
            cou=cou-1
            # print(i,"cut")
            continue
        # print(i,"save")
        ind_end = i + n
        if ind_end > len(Seq):
            break
        T1 = Seq[i:ind_end, :-1]
        T2 = Seq[ind_end-1, -1]
        if ind_end%30==0:
            X3.append(T1)
            y3.append(T2)
            continue
        if ind_end%30==29:
            X2.append(T1)
            y2.append(T2)
            continue
        if ind_end%30==28:
            X4.append(T1)
            y4.append(T2)
            continue
        if ind_end%30==25 or ind_end%30==20:
            X1.append(T1)
            y1.append(T2)
            continue
        X.append(T1)
        y.append(T2)
    return array(X), array(y), array(X1), array(y1), array(X2), array(y2), array(X3), array(y3), array(X4), array(y4)

n_steps =9
n_features = 126

X, y , X1 , y1, X2, y2, X3, y3, X4, y4 = split_seq(seq,n_steps)

def get_compiled_catch():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=63,
                                kernel_size=(5,),
                                activation='relu',
                                padding  = 'same',
                                kernel_regularizer=regularizers.l2(0.001),
                                input_shape=(n_steps, n_features)),
        tf.keras.layers.Conv1D(filters=63,
                                kernel_size=(5,),
                                activation='relu',
                                padding  = 'same',
                                kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        tf.keras.layers.Conv1D(filters=63,
                                kernel_size=(5,),
                                activation='relu',
                                padding  = 'same',
                                kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Conv1D(filters=63,
                                kernel_size=(5,),
                                activation='relu',
                                padding  = 'same',
                                kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        tf.keras.layers.Conv1D(filters=63,
                                kernel_size=(5,),
                                activation='relu',
                                padding  = 'same',
                                kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Conv1D(filters=63,
                                kernel_size=(5,),
                                activation='relu',
                                padding  = 'same',
                                kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        tf.keras.layers.LSTM(45, activation='relu',kernel_regularizer=regularizers.l2(0.001),return_sequences=False),
        layers.Dropout(0.3),
        tf.keras.layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.001),activation='relu'),
        layers.Dropout(0.3),
        tf.keras.layers.Dense(units=1,kernel_regularizer=regularizers.l2(0.001))
    ])
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                  optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False),
                  metrics=['RootMeanSquaredError'])

    return model

start = time.perf_counter()
catch=get_compiled_catch()
history = catch.fit(X, y ,validation_data=(X1, y1),batch_size=(30-n_steps+1-5),epochs=50)

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('value')
plt.xlabel('epoch')
plt.legend(['train-rmsR=e', 'val-rmse','train-loss','val-loss'], loc='upper right')
plt.show()
Y1=catch.predict(X4)
_ ,rmse=catch.evaluate(X4,y4)
print("2016 rmse=",rmse)
Y2=catch.predict(X2)
_ ,rmse=catch.evaluate(X2,y2)
print("2017 rmse=",rmse)
Y3=catch.predict(X3)
_ ,rmse=catch.evaluate(X3,y3)
print("2018 rmse=",rmse)
end = time.perf_counter()

print(end - start)

