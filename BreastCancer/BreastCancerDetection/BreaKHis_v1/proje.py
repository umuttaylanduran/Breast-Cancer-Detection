# Yapay zeka modelimizi geliştirmek, Resnet101'den faydalanmak ve Data Augmentation işlemleri için.
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Dataframe oluşturma, ihtiyacımız olmayan verilerden arınmak için.
import pandas as pd
import os

# Grafik oluşturma vs. işlemler için
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# Resim boyutu
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)

# Model oluşturma
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))  # 4 farklı kanser türü için

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Veriyi yükleme
df = pd.read_csv('Folds.csv')
df = df[df['mag'] == 100] # Yalnızca 100X olanları kullanacağız.
df = df[~df['filename'].str.contains('benign')] # benign olanları çıkarıyoruz çünkü yalnızca "malignant" kullanılacak.

df['class'] = df['filename'].apply(lambda x: x.split('/')[5]) # filename'den etiketleri al
df['filename'] = df['filename'].apply(lambda x: x if os.path.isfile(x) else "None") # Eğer dosya yoksa, "None" yap

train_df = df[df['grp'].str.contains('train')] # train verisetlerini ayırıyoruz.
test_df = df[df['grp'].str.contains('test')] # test verisetlerini ayırıyoruz.

# Data augmentation
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="filename",
    y_col="class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(img_width,img_height))

valid_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="filename",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(img_width,img_height))

# Modeli eğitme
history = model.fit(train_generator, validation_data = valid_generator, epochs = 10)

# Eğitim ve doğrulama doğrulukları
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print("Eğitim Doğruluk: ", train_acc[-1])
print("Doğrulama Doğruluk: ", val_acc[-1])

# Çıkan sonuçlardan grafik oluşturma:
# 1) Eğitim ve Doğrulama Doğrulukları Grafiği --> Modelin overfitting olup olmadığını gözlemlemek için bu grafikten faydalanabiliriz.
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('accuracy_graph.png')  # Grafik dosyasını kaydeder
plt.show()

# 2) Eğitim ve Doğrulama Kayıpları Grafiği (loss) --> Her sinir ağı (epoch) için eğitim ve doğrulama kayıplarını gösterir.
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss_graph.png')  # Grafik dosyasını kaydeder
plt.show()

# 3) Confusion Matrix (Karışıklık Matrisi) --> Modelin hangi sınıfları doğru tahmin ettiğini, hangilerinde hata yaptığını gösterir.
# Confusion Matrix
pred = model.predict(valid_generator)
pred_classes = np.argmax(pred, axis=1)

cm = confusion_matrix(valid_generator.classes, pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"])
disp.plot()
plt.savefig('confusion_matrix.png')  # Grafik dosyasını kaydeder
