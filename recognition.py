import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

X_train_v2 = X_train[:1000, :]
y_train_v2 = y_train[:1000]
X_test = X_test[:100, :]
y_test = y_test[:100]
 
model_v2 = svm.SVC()
model_v2.fit(X_train_v2, y_train_v2)

y_pred_v2 = model_v2.predict(X_test)
import matplotlib.pyplot as plt

indexToCompare = 0
plt.figure(figsize=(10, 10))  # Set the figure size for the entire grid

for i in range(0, 9):
    indexToCompare = indexToCompare + 1
    
    title = 'True: ' + str(y_test[indexToCompare]) + ', Prediction: ' + str(y_pred_v2[indexToCompare])
    
    plt.subplot(3, 3, i+1)  # 3 rows, 3 columns, i+1 is the index of the current subplot
    plt.title(title)
    plt.imshow(X_test[indexToCompare].reshape(28,28), cmap='gray')
    plt.grid(None)
    plt.axis('off')

plt.tight_layout()  # Adjust spacing between subplots for a clean layout
plt.show()

 
acc_v2 = metrics.accuracy_score(y_test, y_pred_v2)
print('\nAccuracy: ', acc_v2)

from sklearn.metrics import ConfusionMatrixDisplay

cm = metrics.confusion_matrix(y_test, y_pred_v2,labels=model_v2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_v2.classes_)           
disp.plot()

plt.show()