import matplotlib.pyplot as plt
from le_net import LeNet
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.datasets import mnist
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("leNet")

if __name__ == "__main__":    
    # file path of our model
    model_path_name = 'lenet_model.keras'
    # to test our model
    (_, _), (x_test, y_test) = mnist.load_data()
    
    x_test_prep = x_test/255.0
    x_test_prep = x_test_prep.reshape(x_test_prep.shape[0],28,28,1)
    
    batchs_epochs = [[32,20],[64,10]]#,[64,40],[128,20]]
    mlflow.tensorflow.autolog()
    
    with mlflow.start_run(run_name='LeNet_hyperparameters_search') as parent_run:
        # Initialize and load model
        for b_e in batchs_epochs:
            with mlflow.start_run(run_name = f'trail_b{b_e[0]}_e{b_e[1]}', nested = True) as child_run:
                mlflow.log_param("batch size", b_e[0])
                mlflow.log_param("epochs", b_e[1])
                network = LeNet(batch_size=b_e[0], epochs=b_e[1])
                network.train()
    
        # metrics = network.evaluate(x_test_prep, y_test)
        # predictions = network.predict()

        