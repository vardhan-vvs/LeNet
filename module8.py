import argparse
import matplotlib.pyplot as plt
from le_net import LeNet
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.datasets import mnist

if __name__ == "__main__":
    # Parse command-line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("image_filename", type=str)
    # parser.add_argument("actual_digit", type=int)
    # args = parser.parse_args()
    
    # file path of our model
    model_path_name = 'lenet_model.keras'
    # to test our model
    (_, _), (x_test, y_test) = mnist.load_data()
    
    x_test_prep = x_test/255.0
    x_test_prep = x_test_prep.reshape(x_test_prep.shape[0],28,28,1)
    
    # Initialize and load model
    network = LeNet(batch_size=64, epochs=20)
    if os.path.exists(f'{model_path_name}'):
        pass
    else:
        network.train()
    
    network.save(model_path_name)
    network.load(model_path_name)
    
    
    # grayscale_image = Image.open(args.image_filename).convert('L')
    # resized_image = grayscale_image.resize((28, 28))  # Resize to 28x28
    # image_data = np.array(resized_image).astype(np.float32)
    # image_data = image_data.flatten() / 255.0  # Normalize
    
    y_hat = network.predict(x_test_prep)
    y_hat_labels = np.argmax(y_hat, axis = 1)
    # image_data = image_data.reshape(1, 28,28)
    # y_hat = network.predict(image_data)
    
    # y_hat = np.argmax(y_hat)
    metrics = network.evaluate(x_test_prep, y_test)
    print(metrics)
    
    # image = plt.imread(args.image_filename)
    # plt.imshow(image, cmap="gray")
    # plt.show()
    
    # Output result
    # if y_hat == args.actual_digit:
        # print(f"Success: Image {args.image_filename} is for digit {args.actual_digit} is recognized as {y_hat}.")
    # else:
    #     print(f"Fail: Image {args.image_filename} is for digit {args.actual_digit} but the inference result is {y_hat}.")
    

    