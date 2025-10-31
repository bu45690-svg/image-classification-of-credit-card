import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers,models
from sklearn.metrics import confusion_matrix,precision_score,f1_score
import numpy as np


train_set=r"C:\Users\DELL\OneDrive\Documents\Project\credit\train"
test_set=r"C:\Users\DELL\OneDrive\Documents\Project\credit\test"

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_data=train_datagen.flow_from_directory(
    train_set,
    target_size=(128,128),
    batch_size=8,
    class_mode='categorical'
)
test_data=test_datagen.flow_from_directory(
    test_set,
    target_size=(128,128),
    batch_size=8,
    class_mode='categorical'
)
print(train_data.image_shape)
print(len(train_data))
print(len(test_data))

model=models.Sequential([
    layers.Input(shape=(128,128,3)),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(3, activation='softmax')


])

model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
history = model.fit( train_data, epochs=15, validation_data=test_data )
loss, acc = model.evaluate(test_data)
print("Test Accuracy:", acc)
print("Test Accuracy:", acc * 100, "%")
print("Test loss:", loss)


Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)


y_true=test_data.classes
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print('precision:',precision_score(y_true,y_pred,average='macro'))
print('f1_score:',f1_score(y_true,y_pred,average='macro'))

# i take a image to predict that it give the correct image classify or not i use a visa credit card image

img_path = r"C:\Users\DELL\OneDrive\Documents\Project\credit\test\Visa\00115269-9145-4fe4-b5c9-c7977bdae9d7.png"

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
labels = list(train_data.class_indices.keys())
print("Predicted Class:", labels[predicted_class[0]])
