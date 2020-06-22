print("[INFO] Importing Libraries")
import matplotlib as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time   # time1 = time.time(); print('Time taken: {:.1f} seconds'.format(time.time() - time1))
import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from PIL import Image 
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras_applications.resnet import ResNet50
from keras_applications.mobilenet import MobileNet
SEED = 50   # set random seed
print("[INFO] Libraries Imported")

from keras.applications.vgg16 import VGG16

from keras.utils import plot_model

#%%   
def make_model():

    print("[INFO] Compiling Model...")
    
    #base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(250,250,3), pooling=None, classes=11)
    #base_model = keras.applications.resnet_v2.ResNet152V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=3)
    base_model = keras.applications.mobilenet.MobileNet(input_shape=(200,266,3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=7)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    
    # #for layer in base_model.layers:
    #     #print (layer.name)
    
    
    #plot_model(base_model, to_file='basevgg16.png')
    for layer in base_model.layers:
          layer.trainable = True
    #for layer in base_model.layers[31:]: #block8_7_ac
           #layer.trainable = True
    # #base_model.summary()
    
    x = layer_dict['conv_pw_13_relu'].output
    x= GlobalAveragePooling2D()(x)
    #x = Flatten()(x)
    # x = Dense(750, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(2000, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    
    
    x = Dense(7, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    
    
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod1.png')
    print("[INFO] Model Compiled!")
    return model
  
#%%

def load_images():  
    print("[INFO] loading images from folders")
    data = []
    labels = []
    labels2 = []
    
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images('C:\\Users\\User\\COVID 19 THIRD FINAL\\DSET FINAL MULTI')))   # data folder with 2 categorical folders
    random.seed(SEED)
    random.shuffle(imagePaths)
    
    
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), 
        # flatten the 32x32x3=3072 pixel image into a list, and store the image in the data list
        
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (266, 200))/255
        data.append(image)
     
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2].split("_")
        labels.append(label)
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float")
    labels = np.array(labels)
    Labels_verbal = labels
    
    
    
    print("[INFO] Private data images loaded!")
    
    print("[INFO] Reshaping data!")
    
    #data = data.reshape(data.shape[0], 400, 300, 3)
    
    print("[INFO] Data Reshaped to feed into models channels last")
    
    from sklearn.preprocessing import MultiLabelBinarizer
    print("[INFO] Labels formatting")
    lb = MultiLabelBinarizer()
    labels = lb.fit_transform(labels) 
    #labels = keras.utils.to_categorical(labels, num_classes=3, dtype='float32')
    print("[INFO] Labels ok!")
    print("Dataset is ready")
    return data,labels,Labels_verbal

#%%
data,labels,Labels_verbal=load_images()

#%%
time1 = time.time() #initiate time counter
n_split=10 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces

omega = 1

for train_index,test_index in KFold(n_split).split(data):
    trainX,testX=data[train_index],data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    


    model3 = make_model() #in every iteration we retrain the model from the start and not from where it stopped
    if omega == 1:
       model3.summary()
    omega = omega + 1   
    
    print('[INFO] PREPARING FOLD: '+str(omega-1))
    model3.fit(trainX, trainY,epochs=8, batch_size=64)
    
    #aug = ImageDataGenerator(rotation_range=10, width_shift_range=2, height_shift_range=2)
    #aug.fit(trainX)
    #model3.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=8, steps_per_epoch=len(trainX)//64)
    score = model3.evaluate(testX,testY)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',model3.evaluate(testX,testY))
    
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    #conf = confusion_matrix(testY, predict) #get the fold conf matrix
    #conf_final = conf + conf_final #sum it with the previous conf matrix
    #name2 = name2 + 1
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    #predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    testY = testY.argmax(axis=-1)
    test_labels = np.concatenate ([test_labels, testY]) #merge the two np arrays of labels
#scores = np.asarray(scores)
#final_score = np.mean(scores)


print("[INFO] Results Obtained!")
print('Time taken: {:.1f} seconds'.format(time.time() - time1)) 




