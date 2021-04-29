import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from keras_preprocessing import image
import cv2
#%% Train klasöründeki kayıtları Okur sayılar listesine demet olarak sayıların sınıfını ve yolunu yazar.
number_types = os.listdir('Images\\mnist-train') ## mnist-train dosyanın içindeki klasörlerin isimlerini number_typesa yazar{0,1,2,3,4,5,6,7,8,9}


train_numbers = []
for number_type in number_types:
 all_numbers = os.listdir(f"Images\\mnist-train\\{number_type}")
 
 for number in all_numbers:
     train_numbers.append((number_type,f"Images\\mnist-train\\{number_type}\\{number}"))
#örnek train_numbers[0] = ('0', 'Images\\mnist-train\\0\\1.png') 
#%% Aynısını Test İçin yapıyoruz.
test_numbers = []
for number_type in number_types:
 all_numbers = os.listdir(f"Images\\mnist-test\\{number_type}")
 
 for number in all_numbers:
     test_numbers.append((number_type,f"Images\\mnist-test\\{number_type}\\{number}"))
#%%Numbers arrayini dataframe'e çevirdik.
train_numbers_df = pd.DataFrame(data=train_numbers, columns=['class', 'image_path'])
test_numbers_df = pd.DataFrame(data=test_numbers,columns = ['class','image_path'])

#%%Sınıf labelına Label Encoding uygulanıyor.Bizim örneğimizde label encodinge gerek yok labellar zaten 0'dan 9'a kadar rakamlardan oluştuğu için. Fakat yine de protokol gereği uygulandı. 
from sklearn.preprocessing import LabelEncoder
train_label = train_numbers_df["class"].values
test_label = test_numbers_df["class"].values
label_encoder = LabelEncoder()
train_label = label_encoder.fit_transform(train_label)
test_label = label_encoder.fit_transform(test_label)
#%% train resimlerini okuyup sayısal değerlere çevirdik.#train_images arrayinin her bir elemanı (28,28,3) boyutunda bir liste. 28x28 pixel ve her bir pixel için rgb(255,255,255) değerleri.
image_paths = train_numbers_df['image_path'].values
train_images = []
for path in image_paths:
    img = cv2.imread(path)
    img = cv2.resize(img,(28,28))
    train_images.append(img)
train_images = np.array(train_images)
train_images = train_images.astype('float32')/255.0 ## Normalizasyon uygulandı. RGB değerleri maksimum 255 minimum 0 değeri alabileceği için 255'e bölündü.(Min-Max Normalizasyonu)

#%% test resimlerini okuyup sayısal değere çevirdik.
image_paths = test_numbers_df['image_path'].values
test_images = []
for path in image_paths:
    img = cv2.imread(path)
    img = cv2.resize(img,(28,28))
    test_images.append(img)
test_images = np.array(test_images)
test_images = test_images.astype('float32')/255.0
#%%Yapay Sinir Ağı Modeli Oluşturuldu.
import keras
import tensorflow as tf
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28,3)),#input katmanı
    keras.layers.Dense(units=300, activation='relu'),#gizli katman relu 300 node bulunuyor.
    keras.layers.Dense(units=10, activation=tf.nn.softmax)#output katmanı
])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),loss='sparse_categorical_crossentropy')#0.01 öğrenme hızıyla Adam gradyan yöntemi kullanıldı.
model.fit(train_images,train_label, epochs=6)
#%%Train setinin Accuracy hesabı
train_predicts = model.predict_classes(train_images)
train_true_counts = {item : 0 for item in number_types}
index = 0
for i in range(0,len(train_images)):
    actual_class = train_numbers_df['class'].values[i]
    class_item_count = train_numbers_df['class'].value_counts()[str(actual_class)]
    if(str(train_predicts[i]) == actual_class):
        train_true_counts[str(actual_class)] += 1
train_class_accuracy = {i:train_true_counts[str(i)]/train_numbers_df['class'].value_counts()[str(i)] for i in number_types}
#%%Test setinin Accuracy hesabı
test_predicts = model.predict_classes(test_images)
test_true_counts = {item : 0 for item in number_types}
index = 0
for i in range(0,len(test_images)):
    actual_class = test_numbers_df['class'].values[i]
    class_item_count = test_numbers_df['class'].value_counts()[str(actual_class)]
    if(str(test_predicts[i]) == actual_class):
        test_true_counts[str(actual_class)] += 1
test_class_accuracy = {i:test_true_counts[str(i)]/test_numbers_df['class'].value_counts()[str(i)] for i in number_types}

#%%Accuracy tablolarının plot olarak çizdirilmesi

fig, ax =plt.subplots(1,1)
train_accuracy_list=[train_class_accuracy[str(i)] for i in range(0,len(train_class_accuracy))]
test_accuracy_list =[test_class_accuracy[str(i)] for i in range(0,len(test_class_accuracy))]
train_accuracy_sum = 0
test_accuracy_sum = 0
for i in range(0,len(train_accuracy_list)):
    train_accuracy_sum += train_accuracy_list[i]
    test_accuracy_sum += test_accuracy_list[i]
train_accuracy_avg = float(np.round(train_accuracy_sum/len(train_accuracy_list),2))
test_accuracy_avg = float(np.round(test_accuracy_sum/len(test_accuracy_list),2))
train_accuracy_list.append(train_accuracy_avg)
test_accuracy_list.append(test_accuracy_avg)
print("~~Train Accuracy Hesabı~~")
print(train_class_accuracy,end="")
print(f" AVG:{train_accuracy_avg}")
print("\n~~Test Accuracy Hesabı~~")
print(test_class_accuracy,end="")
print(f" AVG:{test_accuracy_avg}")
value_list = train_accuracy_list,test_accuracy_list
data=value_list
column_labels=number_types
column_labels.append("Avg")
row_labels = ["Train Accuracy","Test Accuracy"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center",rowLabels=row_labels)
plt.title("Accuracy Table")
plt.show()
#%%

fig, ax =plt.subplots(1,1)
true_counts = [train_true_counts[str(i)] for i in range(0,len(train_true_counts))]
total_counts = [train_numbers_df['class'].value_counts()[str(i)] for i in range(0,len(train_true_counts))]
value_list = true_counts,total_counts
data= value_list
column_labels=number_types
row_labels = ["True Counts","Total Counts"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center",rowLabels=row_labels)
plt.title("Train True Counts And Total Counts")
plt.show()
#%%

fig, ax =plt.subplots(1,1)
true_counts = [test_true_counts[str(i)] for i in range(0,len(test_true_counts))]
total_counts = [test_numbers_df['class'].value_counts()[str(i)] for i in range(0,len(test_true_counts))]
value_list = true_counts,total_counts
data= value_list
column_labels=number_types
row_labels = ["True Counts","Total Counts"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center",rowLabels=row_labels)
plt.title("Test True Counts And Real Counts")
plt.show()      
        