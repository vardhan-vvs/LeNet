import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

class LeNet:
    def __init__(self, batch_size=32, epochs=20):
        '''
        Initializing this class will create LeNet layers and compiles them.
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()
    

    def _create_lenet(self):
        '''
        This function will create the model along with its layers using keras.
        It takes no arguments.
        It returns nothing.
        '''
        self.model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(filters=6, kernel_size=(5,5), 
                    activation='sigmoid', # input_shape=(28, 28, 1), 
                    padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            
            Conv2D(filters=16, kernel_size=(5,5), 
                    activation='sigmoid', 
                    padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),

            Flatten(),

            Dense(120, activation='sigmoid'),
            Dense(84, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        '''
        This function will compile our model.
        It takes no arguments.
        It returns nothing.
        '''
        if self.model is None:
            print('Error: Create a model first..')
        
        self.model.compile(optimizer='Adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        

    def _preprocess(self):
        '''
        This function will pre process our images from mnist dataset to be suitable
        for testing and training.
        It takes no arguments.
        It returns nothing.
        '''
        # load mnist data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # normalize
        x_train = x_train/255.0
        x_test = x_test/255.0

        # add channel dim
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  

        # one-hot encoding
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        '''
        This function will train our model using keras fit function.
        It will first call preprocess function to get images ready for 
        training and then starting to train.
        It takes no arguments
        It returns nothing.
        '''
        self._preprocess()
        self.model.fit(self.x_train, self.y_train, 
                        batch_size=self.batch_size, 
                        epochs=self.epochs,
                        validation_data = (self.x_test, self.y_test))
    
    def save(self, model_path_name):
        '''
        This function will save our model to given directory.
        It takes the directory path as input.
        It returns nothing.
        '''
        if os.path.exists(f'{model_path_name}'):
            pass
        else:
            self.model.save(model_path_name)
    
    def load(self,model_path_name):
        '''
        This function will load our model from given directory.
        It takes the directory path as a parameter.
        It returns nothing.
        '''
        if os.path.exists(f'{model_path_name}'):
            self.model = load_model(model_path_name)
        
    def predict(self,images):
        '''
        This function uses the loaded model and calls predict function with
        the input images.
        It takes a list of images in numpy array format.
        It returns the predicted values to parent or calling function.
        '''
        # predictions = np.argmax(self.loaded_model.predict(images), axis=1)
        predictions = self.model.predict(images)
        return predictions
    
    # def evaluate(self, x_test, y_test):
    #     '''
    #     This function will compute metrics []
    #     It will take two parameters, test images and their labels
    #     It will return a dictionary of metrics
    #     '''
    #     y_pred = self.model.predict(x_test)
    #     y_pred_labels = np.argmax(y_pred, axis=1)
        
    #     accuracy = accuracy_score(y_test, y_pred_labels)
    #     cm = confusion_matrix(y_test, y_pred_labels)
    #     classReport = classification_report(y_test, y_pred_labels)
    #     precision = precision_score(y_test, y_pred_labels, average='macro')
    #     recall = recall_score(y_test, y_pred_labels, average='macro')
    #     f1 = f1_score(y_test, y_pred_labels, average='macro')
        
    #     y_test_onehot = to_categorical(y_test, 10)
    #     auc = roc_auc_score(y_test_onehot, y_pred, multi_class='ovr', average='macro')
        
    #     metrics = {
    #         'accuracy': accuracy,
    #         'precision': precision,
    #         'recall': recall,
    #         'f1 score':f1,
    #         'roc and auc': auc
    #     }
        
    #     print("Confusion matrix:\n", cm)
    #     print("Classification Report:\n", classReport)
        
    #     return metrics