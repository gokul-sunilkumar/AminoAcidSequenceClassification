import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy
from keras.models import Sequential
from keras.layers import Dense , LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)
from os import listdir
from tqdm import tqdm

pd.options.display.float_format = '{:20,.2f}'.format
import os
import seaborn as sns



os.listdir()

os.chdir('./random_split/')

os.listdir()

def read_all_shards(partition='cv'):
    shards = []
    for fn in os.listdir(os.path.join(partition)):
        with open(os.path.join(partition, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    return pd.concat(shards)

test = read_all_shards('test')
cv = read_all_shards('dev')
train = read_all_shards('train')

partitions = {'test': test, 'cv': cv, 'train': train}
for name, df in partitions.items():
    print('"%s" has %d sequences' % (name, len(df)))

# checking for duplicates
print(train.duplicated().any())
print(cv.duplicated().any())
print(test.duplicated().any())

train.head()

#Taking 500k sequences
df_train = train[:500000]
df_cv = cv[:25000]
df_test = test[:25000]
print("df_train: ", df_train.shape)
print("df_cv  : ", df_cv.shape)
print("df_test : ", df_test.shape)

print("Unique classes in train set :", len(np.unique(df_train.family_accession)))
print("Unique classes in cv set :", len(np.unique(df_cv.family_accession)))
print("Unique classes in test set :", len(np.unique(df_test.family_accession)))
print("Unique classes in all the three datasets :",len(set(np.unique(df_test.family_accession)).union(set(np.unique(df_cv.family_accession)),set(np.unique(df_train.family_accession)))))

common_class = set(np.unique(df_test.family_accession)).intersection(set(np.unique(df_cv.family_accession)),set(np.unique(df_train.family_accession)))
print("Common classes :", len(common_class))


df_train = df_train.loc[df_train['family_accession'].isin(common_class)].reset_index()
df_cv = df_cv.loc[df_cv['family_accession'].isin(common_class)].reset_index()
df_test = df_test.loc[df_test['family_accession'].isin(common_class)].reset_index()
print("Shape of our train data : ",df_train.shape)
print("Shape of our cv data : ",df_cv.shape)
print("Shape of our test data : ",df_test.shape)



df_train.head()

partitions = {'test': df_test, 'cv': df_cv, 'train': df_train}
for name, partition in partitions.items():
    partition.groupby('family_accession').size().hist(bins=50)
    plt.title('Distribution of family sizes for %s' % name)
    plt.ylabel('# Families')
    plt.xlabel('Family size')
    plt.show()

plt.figure(figsize=(20,10))
sns.set(style="darkgrid")
ax = sns.countplot(x="family_accession", data=df_train)
plt.show()

#Percentiles
train_des = pd.DataFrame(df_train.sequence.map(len)).describe(include = 'all', percentiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9])
cv_des = pd.DataFrame(df_cv.sequence.map(len)).describe(include = 'all', percentiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9])
test_des = pd.DataFrame(df_test.sequence.map(len)).describe(include = 'all', percentiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9])

y_pos = np.arange(13)
ticks = ['mean','std','min','10%','20%','30%','40%','50%','60%','70%','80%','90%','max']
plt.bar(y_pos ,(train_des[1:].sequence), label='train')
s = plt.xticks(y_pos,ticks)
plt.xlabel('percentile of lengths')
plt.ylabel('length of sequence')

plt.bar(y_pos ,cv_des[1:].sequence,color='y')
plt.xlabel('percentile of lengths')
plt.ylabel('length of sequence')
s = plt.xticks(y_pos,ticks)

plt.bar(y_pos ,test_des[1:].sequence,color='r')
plt.xlabel('percentile of lengths')
plt.ylabel('length of sequence')
s = plt.xticks(y_pos,ticks)

print('Following are the plots of the most frequent family_ids and their counts')
plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
plt.title('Train')
df_train.groupby('family_id').size().sort_values(ascending=False).head(20).plot('bar')
plt.subplot(1,3,2)
plt.title('cv')
df_cv.groupby('family_id').size().sort_values(ascending=False).head(20).plot('bar',color='y')
plt.subplot(1,3,3)
plt.title('Test')
df_test.groupby('family_id').size().sort_values(ascending=False).head(20).plot('bar',color='red')
plt.show()

df_train["family_accession"].value_counts()

def oversample(df):
    classes = df["family_accession"].value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df["family_accession"] == key]) 
    classes_sample = []
    for i in range(1,len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df

def undersample(df):
    classes = df["family_accession"].value_counts().to_dict()
    # least_class_amount = (max(classes.values())/3)
    least_class_amount = 50
    classes_list = []
    for key in classes:
        classes_list.append(df[df["family_accession"] == key]) 
    classes_sample = []
    for i in range(0,len(classes_list)):
        classes_sample.append(classes_list[i].sample(least_class_amount))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[-1]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df

df_sample=df_cv.copy()
df_sample["family_accession"].value_counts()

df_osample=oversample(df_sample)
df_osample["family_accession"].value_counts()

df_usample=undersample(df_osample)
df_usample["family_accession"].value_counts()


def resampling(df):
    classes = df["family_accession"].value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df["family_accession"] == key]) 
    classes_sample = []
    for i in range(1,len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    df = final_df

    classes = df["family_accession"].value_counts().to_dict()
    least_class_amount = (classes.values()/3)
    classes_list = []
    for key in classes:
        classes_list.append(df[df["family_accession"] == key]) 
    classes_sample = []
    for i in range(0,len(classes_list)-1):
        classes_sample.append(classes_list[i].sample(least_class_amount))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[-1]], axis=0)
    final_df = final_df.reset_index(drop=True)

    return final_df

df_train = resampling(df_train)

import time
def plt_dynamic( x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()
    plt.show()

def space_in_sequence(word):
    s_= ''
    for i in range(0, len(word), 1):
        s_ = s_ + ' ' + word[i]
    return s_

def top_freq_features(df, n):
    return df[:n]
space_in_sequence(df_train.sequence[2])

def compile_execute_model():
    print("Model Summary:")
    model.summary()
    print("Model Compilation:")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model Execution:")
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_cv, y_cv))
    score = model.evaluate(X_test, y_test, verbose=0) 
    print('cv score:', score[0]) 
    print('cv accuracy:', score[1])
    
    model_pretty_table.add_row([model_name, round(score[0]*100,2), round(score[1]*100,2)])
    
    fig,ax = plt.subplots(1,1)
    ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

    x = list(range(1,nb_epoch+1))

    vy = history.history['val_loss']
    ty = history.history['loss']
    plt_dynamic(fig, x, vy, ty, ax)

def get_index(word):
    word = word.lower()
    temp = df['feature'] == word
    temp_sum = np.array(temp).sum()
    if(temp_sum ==1):
        return df.index[df['feature'] == word].tolist()[0]
    else:
        return top_words

def right_padding_with_index(x_train, max_seq_length):
    from tqdm import tqdm
    lis =[]
    i = 0
    for j in tqdm(np.arange(0,len(x_train))):
        x = x_train[j]
        x_iter = map(get_index, x.split())
        l1 = list(x_iter)
        lis.append(l1)
    arr = sequence.pad_sequences(lis, maxlen=100, padding='post')
    return arr

from keras.utils import np_utils
def get_categorical(y_train, number_of_unique_classes):
    Y_train = []
    for label in y_train:
        label_ = np_utils.to_categorical(label , number_of_unique_classes) 
        Y_train.append(label_)
    Y_train = np.array(Y_train)
    return Y_train

x_train = (df_train.sequence).apply(space_in_sequence)
x_cv    = (df_cv.sequence).apply(space_in_sequence)
x_test  = (df_test.sequence).apply(space_in_sequence)

values   = np.arange(0,len(df_train.family_accession.unique()),1)
keys = df_train.family_accession.unique()
dict_class = dict(zip(keys, values))

y_train = df_train.family_accession.apply(lambda x: dict_class[x])
y_cv    = df_cv.family_accession.apply(lambda x: dict_class[x])
y_test  = df_test.family_accession.apply(lambda x: dict_class[x])

bow_scalar = CountVectorizer(token_pattern = r"(?u)\b\w+\b")
bow_scalar.fit(x_train.values)

final_counts = bow_scalar.transform(x_train.values).toarray()
final_counts = final_counts.sum(axis=0)
final_counts.shape

final_counts     

df = pd.DataFrame({ 'feature': bow_scalar.get_feature_names(),'frequency': list(final_counts) })
df.sort_values('frequency', ascending = False, inplace = True)
df.reset_index(drop=True, inplace = True)
df.head(25)

top_words = 20       
max_seq_length = 100  
from tqdm import tqdm
df = top_freq_features(df, top_words)

df

unique_indexes = top_words + 1
print("unique_indexes: ", unique_indexes)

os.chdir('../seq_500/')


from tqdm import tqdm
if not os.path.isfile('X_train.csv'):
    print("calculating X_train ...")
    X_train = right_padding_with_index(x_train[:], max_seq_length)     #performing right padding on our data
    pd.DataFrame(X_train).to_csv("X_train.csv", index=False, mode = 'w', header = True)


if not os.path.isfile('X_cv.csv'):
    print("calculating X_cv ...")
    X_cv = right_padding_with_index(x_cv[:], max_seq_length)
    pd.DataFrame(X_cv).to_csv("X_cv.csv", index=False, mode = 'w', header = True)


if not os.path.isfile('X_test.csv'):
    print("calculating X_test ...")
    X_test  = right_padding_with_index(x_test[:], max_seq_length)
    pd.DataFrame(X_test).to_csv("X_test.csv", index=False, mode = 'w', header = True)


import numpy as np
def one_hot(final_sequence):
    nb_classes = 21                          
    targets = np.array(final_sequence)
    one_hot_train = np.eye(nb_classes)[targets]
    
    return one_hot_train


# One-hot Encoding
X_train        = pd.read_csv("X_train.csv").values
ohe_train  = one_hot(X_train[:])  
X_cv           = pd.read_csv("X_cv.csv").values 
ohe_cv     = one_hot(X_cv[:]) 
X_test        = pd.read_csv("X_test.csv").values 
ohe_test   = one_hot(X_test[:]) 

print("X_train: ", ohe_train.shape); print(ohe_train[0:5])
print("X_cv: ", ohe_cv.shape);       print(ohe_cv[0:5])
print("X_test: ", ohe_test.shape);   print(ohe_test[0:5])

number_of_unique_classes = len(common_class)

nuq=number_of_unique_classes

Y_train = get_categorical(y_train , number_of_unique_classes)      
Y_cv = get_categorical(y_cv , number_of_unique_classes)
Y_test = get_categorical(y_test , number_of_unique_classes)

np.save('Y_train',Y_train)
np.save('Y_cv',Y_cv)
np.save('Y_test',Y_test)



Y_train = np.load('Y_train.npy')
Y_cv = np.load('Y_cv.npy')
Y_test = np.load('Y_test.npy')


from keras.layers import BatchNormalization,Dropout,Conv1D,Activation,Add,Flatten,Dense
from keras.layers import MaxPooling1D
from keras.layers.merge import concatenate
from keras.initializers import glorot_uniform
from keras.layers import ZeroPadding1D
from keras.models import Input,Model

import warnings 
warnings.filterwarnings('ignore')

input_seq = Input(shape=(100,21))
print(input_seq.shape)

c1= Conv1D(32, 1 , strides=1,padding='valid', name='conv1d_1', kernel_initializer=glorot_uniform(seed=0))(input_seq)

m1  = MaxPooling1D(pool_size=2)(c1)
b1 = BatchNormalization(axis=2, name='batch_normalization_1')(m1)
a1 = Activation('relu',name='activation_1')(b1)

b2 = BatchNormalization(axis=2, name='batch_normalization_2')(a1)
a2 = Activation('relu',name='activation_2')(b2)

c2 = Conv1D(128, 1 , strides=1,padding='valid', name='conv1d_3', kernel_initializer=glorot_uniform(seed=0))(a2)

b3 = BatchNormalization(axis=2, name='batch_normalization_3')(c2)
a3 = Activation('relu',name='activation_3')(b3)

c4 = Conv1D(128 , 1 , strides=1 ,padding='valid', name='conv1d_4' , kernel_initializer=glorot_uniform(seed=0))(a3)

d1 = Dropout(0.5,name='d3')(c4)
m2 = MaxPooling1D(pool_size=2)(d1)

c5 = Conv1D(128,  1 , strides=1 ,padding ='valid',name='conv1d_2',  kernel_initializer=glorot_uniform(seed=0))(m2)

d2 = Dropout(0.5,name='d7')(c5)
m3 = MaxPooling1D(pool_size=2)(d2)

#X10 = Add()([X8,X9])

a4 = Activation('relu',name='activation_4')(m3)
d3 = Dropout(0.2)(a4)
b4 = BatchNormalization(axis=2,name='batch_normalization_4')(d3)
a5 = Activation('relu',name='activation_5')(b4)
d4 = Dropout(0.5,name='dropout_1')(a5)

l1 = LSTM(256)(d4)
f1 = Flatten(name='flatten_1')(l1)

de1 = Dense(nuq ,name='fc' + str(nuq), kernel_initializer = glorot_uniform(seed=0))(f1)
a6 = Activation('softmax',name='activation_6')(de1)

model = Model(inputs = input_seq, outputs = X17)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

ohe_train.shape

history=model.fit(ohe_train, Y_train, epochs=50, batch_size=256 , validation_data=(ohe_cv,Y_cv), verbose=1)

score = model.evaluate(ohe_test, Y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')


x = list(range(1,50+1))

vy = history.history['val_loss']
ty = history.history['loss']


model.save('model.h5')

np.save('history',history)

history = np.load('history.npy',allow_pickle=True)

from keras.models import load_model

model = load_model('model.h5')

score = model.evaluate(ohe_test, Y_test , verbose=1)

print("Test loss:",score[0])
print("Test accuracy:",score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

from prettytable import PrettyTable

x = PrettyTable(['Model','epochs','test loss','test accuracy'])
x.add_row(['Deep CNN',50,score[0],score[1]])
print(x)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(ohe_test)

import numpy as np
y_test_arg=np.argmax(Y_test,axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print('Classification Report')
print(classification_report(y_test_arg, np.argmax(y_pred, axis=1)))

def pred(test_seq):
  test_seq=[test_seq]
  df2=pd.DataFrame(test_seq)
  df2.columns=['sequence']
  test2  = (df2.sequence).apply(space_in_sequence)
  Test2  = right_padding_with_index(test2[:], max_seq_length)
  ohe_test2   = one_hot(Test2[:])
  y_pred2 = model.predict(ohe_test2)
  c=np.argmax(y_pred2,axis=1)
  print()
  print("Predicted Protein Family for the inputted sequence: ",keys[c])

  pred("HWLQMRDSMNTYNNMVNRCFATCIRSFQEKKVNAEEMDCTKRCVTKFVGYSQRVALRFAE")

