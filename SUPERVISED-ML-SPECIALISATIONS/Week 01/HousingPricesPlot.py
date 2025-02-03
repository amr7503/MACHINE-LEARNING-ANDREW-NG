import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]     # m is the size of array x containing examples
    f_wb = np.zeros(m)   # f_wb is initialised as an array of zeroes with same length as x
    for i in range(m):
        f_wb[i] = w * x[i] + b       # iterates over each example in x and for each example it computes the linear model's output 
        
    return f_wb 

tmp_f_wb = compute_model_output(x_train, w=200, b=100)  
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction') #model prediction as line
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')  #Actual values as scaatter

plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

plt.legend() 
plt.show() 
