import argparse
import matplotlib.pyplot as plt
from le_net import LeNet
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    
    mlflow.tensorflow.autolog()
    
    with mlflow.start_run(run_name='testing') as run:
        # Initialize and load model
        network = LeNet(batch_size=64, epochs=20)
        if os.path.exists(f'{model_path_name}'):
            pass
        else:
            network.train()
    
        metrics = network.evaluate(x_test_prep, y_test)
        
        mlflow.log_metrics({
            # 'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1 score':metrics['f1 score'],
            'roc and auc': metrics['roc and auc']
        })
        