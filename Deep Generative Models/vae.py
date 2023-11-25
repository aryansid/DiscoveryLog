from tensorflow.keras import datasets
import numpy 

def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

if __name__ == "__main__": 
  (x_train,y_train), (x_test,y_test) = datasets.fashion_mnist.load_data()
  x_train = preprocess(x_train)
  x_test = preprocess(x_test)