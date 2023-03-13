# diabetes dataset using linear-regression

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets,linear_model
# from sklearn.metrics import mean_squared_error


# disease=datasets.load_diabetes()
# X=disease.data[:,np.newaxis,2]
# Y=disease.target
# X_train=X[:-30]
# X_test=X[-20:]
# Y_train=Y[:-30]
# Y_test=Y[-20:]



# reg=linear_model.LinearRegression()

# reg.fit(X_train,Y_train)

# Y_predict=reg.predict(X_test)

# accuracy=mean_squared_error(Y_test,Y_predict)

# print(accuracy)

# weights=reg.coef_
# intercept=reg.intercept_
# print(weights,intercept)
# plt.scatter(X_test,Y_test)
# plt.plot(X_test,Y_predict)
# plt.show()



#cars dataset
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

cars=pd.read_csv("CarPrice_Assignment.csv")
# plt.figure(figsize=(16,8))
# plt.scatter(cars['horsepower'],cars['price'],c='black')
# plt.xlabel('Horsepower')
# plt.ylabel('Price')
# plt.show()


X=cars['horsepower'].values.reshape(-1,1)
Y=cars['price'].values.reshape(-1,1)

# re-shaping is done to avoid errors

reg=LinearRegression()
reg.fit(X,Y)
print(reg.coef_[0][0])
print(reg.intercept_[0])
predictions=reg.predict(X) 
# plt.figure(figsize=(16,8))
plt.scatter(cars['horsepower'],cars['price'],c='black')
plt.plot(cars['horsepower'],predictions,c='blue',linewidth=2)
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.show()