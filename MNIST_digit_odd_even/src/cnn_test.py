import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import cnn
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


model_fun = cnn.cnn_model_fn

imgW, imgH, channels = 28, 28, 1
train_epochs = 5
batch_size = 200
drop_rate = 0.4
learn_rate = 0.001
cnn_layers = 2
model_dir = '../models/cnn_model/'
save_summary_frequency = 1


initializers = {
        'fast_conv': tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True),
        'he_rec': tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
        'xavier': tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
    }

parameters = {'img_size': [imgH, imgW, channels], 'summary_steps': save_summary_frequency,
                  'model_dir': model_dir, 'learn_rate': learn_rate, 'drop_rate': drop_rate,
                  'w_inits': initializers, 'depth': cnn_layers}


image_classifier = tf.estimator.Estimator(model_fn=model_fun, model_dir=model_dir, params=parameters)


while True:
        userInput = input("\n Enter filename or q to quit\n")
        if userInput == "q":
            break
        else:
            filename = userInput
            print("Processing..\n")
            #prediction data
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            cv2.namedWindow("Image-window", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Image-window',image)
            # cv2.waitKey(0)
            #resize image
            image_1 = cv2.resize(image,(28,28))
            image_blur = cv2.GaussianBlur(image_1,(5,5),0)
            image_thres = cv2.adaptiveThreshold(image_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            cv2.imshow('Image-window',image)
            cv2.imshow('Image-window-1',image_thres)
            cv2.waitKey(0)
            flat = image_thres.flatten().reshape(1, 784) / 255.0
            image_flat = np.zeros((1, 784))
            image_flat = flat 
            Image_float = image_flat.astype(np.float32)

            predict_data = Image_float
            cv2.destroyAllWindows()
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": predict_data},
                y=None,
                batch_size=1,
                num_epochs=1,
                shuffle=False,
                num_threads=1,
            )

            predictions = image_classifier.predict(predict_input_fn)

            for idx, prediction in enumerate(predictions):
                if prediction["classes"] == 0:
                    print("\n###########################################\n")
                    print(" It is an even number \n")
                    print("###########################################\n")
                else:
                    print("###########################################\n")
                    print("It is an odd number \n")
                    print("###########################################\n")
                    
                    

