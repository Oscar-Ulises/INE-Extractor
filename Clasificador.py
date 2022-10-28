from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np


classifier = Sequential()
classifier.add(Conv2D(filters = 32,kernel_size = (3,3),input_shape = (106,168,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128,activation = 'relu'))
classifier.add(Dense(units = 1,activation = 'sigmoid'))
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_dataset = train_datagen.flow_from_directory('training_path',target_size = (106,168),batch_size = 6,class_mode='binary')
test_dataset = test_datagen.flow_from_directory('testing_path',target_size = (106,168),batch_size = 6,class_mode='binary')

classifier.fit_generator(train_dataset,steps_per_epoch = 18//6,epochs = 25,validation_data = test_dataset,validation_steps = 18//6)




def preprocessing(img):
    img = image.load_img(img, target_size = (106,168,3))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    return img

def INE_MASTER(cara,reverso): 
    img = preprocessing(reverso)
    result = classifier.predict(img)
    if result[0][0] == 0:
       try:
           return ine.INE_D_Data(cara),ine.INE_D(reverso)
       except:
           print("Intente tomar de nuevo la foto de su creedencial de elector.")
    else:
        try:
            return ine.INE_E_Data(cara),ine.INE_E(reverso)
        except:
            print("Intente tomar de nuevo la foto de su creedencial de elector.")
        
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

