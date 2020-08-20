#get all sub directories

import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imutils

def get_all_subdirs(base_dir):

    base_dir = '/content/chapter05-fine_tuning/dataset/'
    subdirs = [os.path.join(base_dir, o) for o in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,o))]
    return subdirs
def rename_image_file(subdirs):
    os.getcwd()
    for subdir in subdirs:
      filelocation = subdir +'/'
      for i, filename in enumerate(os.listdir(filelocation)):
          os.rename(filelocation + filename, filelocation + str(i) + ".jpg")
    return "Done"    

def predict_image(imagepath, labels,label_list,labelDict,model,size=224):
    
    image = cv2.imread(imagepath)
    orig = image.copy()
    # pre-process the image for classification
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    result=(model.predict(image))[0]
    val=np.argmax(model.predict(image), axis=1).tolist()
    res=result[val]*100
    res=res.tolist()
    value = None
    for key in label_list:
    if key in labelDict:
        value = labelDict[key]
        break
    label = str(value)    
    imlabels = "{}: {:.2f}%".format(label, res[0])
    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, imlabels, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)
    # show the output image
    cv2.imwrite(os.cwd()+'/output.png',output)
    return None


def label_quantizer(labels):
    labels = np.array(labels)
    len_of_labels = len(np.unique(labels))
    labelDict = {}
    for i in range(0 , len_of_labels):
        labelDict[i+1] = labels[i]
    label_list = np.unique(labels).tolist()    
    return labelDict  , label_list
