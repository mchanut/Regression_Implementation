import math
import random
import numpy as np
import pandas as pd
import sklearn.model_selection
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# README: all our final code is contained in this .py file. Classes for each model are below, followed by the main function which runs all experiments.
# Code expects the file 'boston.csv' in the directory in which it is running. Wine dataset is taken from ucimlrepo. Plotting code is left commented
# out, but the same plots can be found on the writeup.

class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass

    def process(self, X):
        # Column-wise normalization
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col] / X_copy[col].mean()
        X = X_copy

        return X

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]                         
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])    
        self.w = np.linalg.inv(x.T @ x)@x.T@y
        return self

    def predict(self, x):
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w                             
        return yh

    def mean_squared_error(self, yh, y):
        mse = np.mean((yh-y)**2)                  
        return mse

    def gaussian_basis(self, mu, x):
        gb = GaussianBasis()
        new_feature = gb.gaussian(mu, x)
        return new_feature

class GaussianBasis:
    def gaussian(self, mu, x):
        top = x - mu
        top = np.linalg.norm(top)
        top = top ** 2
        eq = - (top / 2)
        eq = np.exp(eq)
        return eq

class SGDLinearRegression:
    def __init__(self):
        pass

    def set_weights(self, X):
        N,D = X.shape
        self.weights = np.zeros((D+1))

    def process(self, X):
        N,D = X.shape

        # Column-wise normalization
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col] / X_copy[col].mean()
        X = X_copy
        
        X['Biases'] = 1

        return X
    
    def gradient(self, X, y, pred, N):
        diff = pred - y
        dot = np.dot(X.T, diff)
        return dot / N
    
    def gradient_with_Lp_reg(self, X, y, pred, N, lambdaa, p):
        diff = pred - y
        dot = np.dot(X.T, diff)
        grad = dot / N
        grad[1:] += lambdaa * (np.sum(np.abs(self.weights) ** p) ** (1/p))
        return grad
    
    def sgd_minibatch(self, X, y, num_epochs=1000, learning_rate=0.01, batch_size=32):
        """Train the linear regression model using SGD minibatch."""
        n_samples, n_features = X.shape
        if n_samples < batch_size:
            return "Error: batch size larger than dataset size"
        for epoch in range(num_epochs):
            i = random.randint(0, n_samples - batch_size)
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            pred = X_i @ self.weights
            g = self.gradient(X_i, y_i, pred, batch_size)
            self.weights -= g * learning_rate

    def sgd_minibatch_with_momentum(self, X, y, num_epochs=1000, learning_rate=0.01, batch_size=32, momentum_rate=0.9):
        """Train the linear regression model using SGD minibatch."""
        n_samples, n_features = X.shape
        if n_samples < batch_size:
            return "Error: batch size larger than dataset size"
        v = 0
        for epoch in range(num_epochs):
            i = random.randint(0, n_samples - batch_size)
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            pred = X_i @ self.weights
            g = self.gradient(X_i, y_i, pred, batch_size)
            v = momentum_rate * v + learning_rate * g   
            self.weights -=  v

    def sgd_minibatch_with_lp(self, X, y, num_epochs=1000, learning_rate=0.01, batch_size=32, p=1, lambdaa=1):
        """Train the logistic regression model using SGD minibatch."""
        n_samples, n_features = X.shape
        if n_samples < batch_size:
            return "Error: batch size larger than dataset size"
        for epoch in range(num_epochs):
            i = random.randint(0, n_samples - batch_size)
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            pred = X_i @ self.weights
            g = self.gradient_with_Lp_reg(X_i, y_i, pred, batch_size, lambdaa=lambdaa, p=p)
            if np.isnan(g).any(): return
            self.weights -= g * learning_rate

    def fit(self, X, y, num_epochs=1000, learning_rate=0.01, batch_size=32, momentum_rate=None, lambdaa=None, p=None):
        """Train the linear regression model."""
        self.set_weights(X)
        X = self.process(X)
        if momentum_rate:
            self.sgd_minibatch_with_momentum(X, y, num_epochs, learning_rate, batch_size, momentum_rate)
        elif lambdaa and p:
            self.sgd_minibatch_with_lp(X, y, num_epochs, learning_rate, batch_size, lambdaa=lambdaa, p=p)
        else:
            self.sgd_minibatch(X, y, num_epochs, learning_rate, batch_size)

    def predict(self, X):
        X = self.process(X)
        pred = X @ self.weights
        return pred
    
    def mean_squared_error(self, pred, y):
        mse = np.mean((pred-y)**2)
        return mse

class LogisticRegression:

    def __init__(self):
        self.weights = np.zeros((14, 3))

    def process(self, X):
        # Column-wise normalization
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col] / X_copy[col].mean()
        X = X_copy
        
        X['Biases'] = 1

        return X

    def softmax(self, z):
        softmaxes = []
        for value in z:
            max_val = np.max(value)
            maxes = []
            total = 0
            for i in range(len(value)):
                denum = np.exp(value[i] - max_val)
                exponent = value[i] - max_val
                total += denum
            for i in range(len(value)):
                num = np.exp(value[i] - max_val)
                num = num / total
                maxes.append(num)
            softmaxes.append(maxes)
        return softmaxes

    def logit(self, X):
        xs = []
        for index, row in X.iterrows():
            dot = np.dot(self.weights.T, row)
            for value in dot:
                value += dot
            xs.append(dot)
        return xs

    def one_hot(self, y):
        one_hot_encoded = pd.get_dummies(y['class'], prefix='class')
        # one_hot_encoded = pd.get_dummies(y)
        one_hot_array = one_hot_encoded.values.astype(int)
        return one_hot_array

    def CEL(self, y_ohe, z):
        N = z.shape[0]
        ce = -np.sum(y_ohe*np.log(z))/N
        return ce
    
    def gradient(self, X, y, pred, N):
        diff = pred - y
        dot = np.dot(X.T, diff)
        return dot / N
    
    def gradient_with_Lp_reg(self, X, y, pred, N, lambdaa, p):
        diff = pred - y
        dot = np.dot(X.T, diff)
        grad = dot / N
        grad[1:] += lambdaa * (np.sum(np.abs(self.weights) ** p) ** (1/p))
        return grad
    
    def gradient_descent(self, X, y_ohe, num_epochs, learning_rate):
        N,D = X.shape
        for epoch in range(num_epochs):
            z = self.logit(X)
            softmax = self.softmax(z)
            g = self.gradient(X, y_ohe, softmax, N)
            self.weights -= g * learning_rate

    def gradient_descent_with_momentum(self, X, y_ohe, num_epochs, learning_rate, momentum_rate):
        N,D = X.shape
        v = 0
        for epoch in range(num_epochs):
            z = self.logit(X)
            softmax = self.softmax(z)
            g = self.gradient(X, y_ohe, softmax, N)
            v = momentum_rate * v + learning_rate * g   
            self.weights -=  v
       
    def sgd_minibatch(self, X, y, num_epochs=1000, learning_rate=0.01, batch_size=32):
        """Train the logistic regression model using SGD minibatch."""
        n_samples, n_features = X.shape
        if n_samples < batch_size:
            return "Error: batch size larger than dataset size"
        for epoch in range(num_epochs):
            i = random.randint(0, n_samples - batch_size)
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            z = self.logit(X_i)
            softmax = self.softmax(z)
            g = self.gradient(X_i, y_i, softmax, batch_size)
            if np.isnan(g).any(): return
            self.weights -= g * learning_rate

    def sgd_minibatch_with_lp(self, X, y, num_epochs=1000, learning_rate=0.01, batch_size=32, p=1, lambdaa=1):
        """Train the logistic regression model using SGD minibatch."""
        n_samples, n_features = X.shape
        if n_samples < batch_size:
            return "Error: batch size larger than dataset size"
        for epoch in range(num_epochs):
            i = random.randint(0, n_samples - batch_size)
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            z = self.logit(X_i)
            softmax = self.softmax(z)
            g = self.gradient_with_Lp_reg(X_i, y_i, softmax, batch_size, lambdaa=lambdaa, p=p)
            if np.isnan(g).any(): return
            self.weights -= g * learning_rate

    def sgd_minibatch_with_momentum(self, X, y, num_epochs=1000, learning_rate=0.01, batch_size=32, momentum_rate = 0.9):
        """Train the logistic regression model using SGD minibatch."""
        n_samples, n_features = X.shape
        if n_samples < batch_size:
            return "Error: batch size larger than dataset size"
        v = 0
        for epoch in range(num_epochs):
            i = random.randint(0, n_samples - batch_size)
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            z = self.logit(X_i)
            softmax = self.softmax(z)
            g = self.gradient(X_i, y_i, softmax, batch_size)
            if np.isnan(g).any(): return
            v = momentum_rate * v + learning_rate * g   
            self.weights -=  v
        


    def fit(self, X, y, num_epochs=1000, learning_rate=0.01, batch_size=None, momentum_rate=None, lambdaa=None, p=None):
        """Train the logistic regression model."""
        X = self.process(X)
        n_samples, n_features = X.shape
        y_ohe = self.one_hot(y)
        if batch_size == None:
            if momentum_rate == None:
                self.gradient_descent(X, y_ohe, num_epochs, learning_rate)
            else:
                self.gradient_descent_with_momentum(X, y_ohe, num_epochs, learning_rate, momentum_rate)
        elif lambdaa and p:
            self.sgd_minibatch_with_lp(X, y, num_epochs, learning_rate, batch_size, lambdaa=lambdaa, p=p)
        else:
            if momentum_rate == None:
                self.sgd_minibatch(X, y_ohe, num_epochs, learning_rate, batch_size)
            else:
                self.sgd_minibatch_with_momentum(X, y_ohe, num_epochs, learning_rate, batch_size, momentum_rate)

    def predict(self, X):
        X = self.process(X)
        y_pred = []
        z = self.logit(X)
        df = pd.DataFrame(z)
        softmax = self.softmax(z)
        for value in softmax:
            y_pred.append(np.argmax(value) + 1)
        return y_pred

    def accuracy(self, y, pred):
        accuracy = 0
        for i in range(len(y)):
            if y['class'].values[i] == pred[i]:
                accuracy += 1
        accuracy = accuracy / len(y)
        return accuracy
    
    def precision(self, y, pred, wineclass=1):
        precision = 0
        total = 0
        for i in range(len(y)):
            if pred[i] == wineclass:
                total += 1
                if y['class'].values[i] == wineclass:
                    precision += 1
        if total == 0:
            return 0
        else:
            precision = precision / total
            return precision
        
    def recall(self, y, pred, wineclass=1):
        recall = 0
        total = 0
        for i in range(len(y)):
            if y['class'].values[i] == wineclass and pred[i] != wineclass:
                total += 1
            elif pred[i] == wineclass and y['class'].values[i] == wineclass:
                total += 1
                recall +=1
        if total == 0:
            return 0
        else:
            recall = recall / total
            return recall
    def f1(self, y, pred, wineclass=1):
        try:
            return (2 * self.precision(y, pred, wineclass) * self.recall(y, pred, wineclass)) / (self.precision(y, pred, wineclass) + self.recall(y, pred, wineclass))
        except:
            return -1



def main():
    # Set random seed
    random.seed(42)

    # Boston dataset
    boston = pd.read_csv('boston.csv')
    boston = boston.drop('B', axis=1)
    boston = boston.dropna()
    X = boston.drop('MEDV', axis=1)
    y = boston['MEDV']
    X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(X, y, test_size=0.2, random_state=42)

    # fetch dataset
    wine = fetch_ucirepo(id=109)

    # data (as pandas dataframes)
    X = wine.data.features
    y = wine.data.targets

    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X, y, random_state=42)

    print('Computing stats for both datasets')
    print(boston.describe())
    print(wine.data.features.describe())

    # 3.1
    print('---3.1: 80/20 Train Test Split---')

    print('---Dataset 1---')
    linear = LinearRegression()
    sgdlinear = SGDLinearRegression()
    linear.fit(X_train_boston, y_train_boston)
    pred = linear.predict(X_test_boston)
    print(f'Analytical Linear MSE: {linear.mean_squared_error(pred, y_test_boston)}')
    sgdlinear.fit(X_train_boston, y_train_boston)
    pred = sgdlinear.predict(X_test_boston)
    print(f'SGDLinear MSE: {sgdlinear.mean_squared_error(pred, y_test_boston)}')

    print('---Dataset 2---')
    logistic = LogisticRegression()
    logistic.fit(X_train_wine, y_train_wine)
    pred = logistic.predict(X_test_wine)
    print(f'accuracy: {logistic.accuracy(y_test_wine, pred)}')
    for i in range(3):
        print(f'precision class {i + 1}: {logistic.precision(y_test_wine, pred, wineclass=i + 1)}')
        print(f'recall class {i + 1}: {logistic.recall(y_test_wine, pred, wineclass=i + 1)}')
        print(f'f1 class {i + 1}: {logistic.f1(y_test_wine, pred, wineclass=i + 1)}')

    # 3.2
    print('---3.2: 5-Fold Cross Validation---')
    print("---5-Fold Cross Validation: Dataset 1---")
    # 5-Fold Cross Validation on Boston
    fold_size = len(X_train_boston) // 5
    folds_X = []
    folds_y = []
    for i in range(5):
        folds_X.append(X_train_boston[fold_size*i:fold_size*(i+1)])
        folds_y.append(y_train_boston[fold_size*i:fold_size*(i+1)])
    
    # Analytical and SGD Linear
    # https://stackoverflow.com/a/38246298
    total_mse_analytical_train = 0
    total_mse_analytical_test = 0
    total_mse_sgd_train = 0
    total_mse_sgd_test = 0
    for i in range(5):
        linear = LinearRegression()
        sgdlinear = SGDLinearRegression()
        test_X = folds_X[i]
        test_y = folds_y[i]
        train_Xs = []
        train_ys = []
        for j in range(5):
            if j != i:
                train_Xs.append(folds_X[j])
                train_ys.append(folds_y[j])
        train_X = pd.concat(train_Xs)
        train_y = pd.concat(train_ys)
        linear.fit(train_X, train_y)
        pred = linear.predict(test_X)
        total_mse_analytical_train += linear.mean_squared_error(pred, test_y)
        pred_test = linear.predict(X_test_boston)
        total_mse_analytical_test += linear.mean_squared_error(pred_test, y_test_boston)
        sgdlinear.fit(train_X, train_y)
        pred = sgdlinear.predict(test_X)
        total_mse_sgd_train += sgdlinear.mean_squared_error(pred, test_y)
        pred_test = sgdlinear.predict(X_test_boston)
        total_mse_sgd_test += sgdlinear.mean_squared_error(pred_test, y_test_boston)
    print(f'Analytical Linear Regression MSE (Training set): {total_mse_analytical_train / 5}')
    print(f'Analytical Linear Regression MSE (Testing set): {total_mse_analytical_test / 5}')
    print(f'SGD Linear Regression MSE (Training set): {total_mse_sgd_train / 5}')
    print(f'SGD Linear Regression MSE (Testing set): {total_mse_sgd_test / 5}')

    print("---5-Fold Cross Validation: Dataset 2---")
    # 5-Fold Cross Validation on Wine
    fold_size = len(X_train_wine) // 5
    folds_X = []
    folds_y = []
    for i in range(5):
        folds_X.append(X_train_wine[fold_size*i:fold_size*(i+1)])
        folds_y.append(y_train_wine[fold_size*i:fold_size*(i+1)])
    
    # Logistic
    total_accuracy_train = 0
    total_accuracy_test = 0
    for i in range(5):
        logistic = LogisticRegression()
        test_X = folds_X[i]
        test_y = folds_y[i]
        train_Xs = []
        train_ys = []
        for j in range(5):
            if j != i:
                train_Xs.append(folds_X[j])
                train_ys.append(folds_y[j])
        train_X = pd.concat(train_Xs)
        train_y = pd.concat(train_ys)
        logistic.fit(train_X, train_y)
        pred = logistic.predict(test_X)
        total_accuracy_train += logistic.accuracy(test_y, pred)
        pred_test = logistic.predict(X_test_wine)
        total_accuracy_test += logistic.accuracy(y_test_wine, pred_test)
    print(f'Logistic Regression Accuracy (Training set): {total_accuracy_train / 5}')
    print(f'Logistic Regression Accuracy (Testing set): {total_accuracy_test / 5}')

    

    # 3.3
    print('---3.3: Different Test Sizes---')
    test_percents = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    train_points_lin = []
    test_points_lin = []
    train_points_log = []
    test_points_log = []
    for percent in test_percents:
        X_boston = boston.drop('MEDV', axis=1)
        y_boston = boston['MEDV']
        X_wine = wine.data.features
        y_wine = wine.data.targets
        linear = LinearRegression()
        sgdlinear = SGDLinearRegression()
        logistic = LogisticRegression()
        x_train_boston, x_test_boston, y_train_boston, y_test_boston = sklearn.model_selection.train_test_split(X_boston, y_boston, test_size=percent, random_state=42)
        x_train_wine, x_test_wine, y_train_wine, y_test_wine = sklearn.model_selection.train_test_split(X_wine, y_wine, test_size=percent, random_state=42)
        linear.fit(x_train_boston, y_train_boston)
        logistic.fit(x_train_wine, y_train_wine)
        pred_linear1 = linear.predict(x_train_boston)
        pred_linear2 = linear.predict(x_test_boston)
        pred_log1 = logistic.predict(x_train_wine)
        pred_log2 = logistic.predict(x_test_wine)
        mse1 = linear.mean_squared_error(pred_linear1, y_train_boston)
        mse2 = linear.mean_squared_error(pred_linear2, y_test_boston)
        acc1 = logistic.accuracy(y_train_wine, pred_log1)
        acc2 = logistic.accuracy(y_test_wine, pred_log2)
        train_points_lin.append(mse1)
        test_points_lin.append(mse2)
        train_points_log.append(acc1)
        test_points_log.append(acc2)
    
    # figure, axis = pyplot.subplots(1, 2, layout='constrained')
    # axis[0].plot(test_percents, train_points_lin, label='Training')
    # axis[0].plot(test_percents, test_points_lin, label='Testing')
    # axis[0].legend()
    # axis[0].set_title("Linear")
    # axis[1].plot(test_percents, train_points_log, label='Training')
    # axis[1].plot(test_percents, test_points_log, label='Testing')
    # axis[1].legend()
    # axis[1].set_title("Logistic")
    # pyplot.show()
    print(f'Linear (Training): {train_points_lin}')
    print(f'Linear (Testing): {test_points_lin}')
    print(f'Logistic (Training): {train_points_log}')
    print(f'Logistic (Testing): {test_points_log}')


    # 3.4
    print('---3.4: Growing Minibatch Sizes---')
    minibatch_sizes = [8, 16, 32, 64, 128]
    epochs = [100, 500, 1000, 1500, 2000, 2500, 3000]
    results = {}
    results_log = {}

    X_boston = boston.drop('MEDV', axis=1)
    y_boston = boston['MEDV']

    X_wine = wine.data.features
    y_wine = wine.data.targets

    X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

    for epoch_lim in epochs:
        for batch in minibatch_sizes:
            if batch not in results: results[batch] = []
            sgdlinear = SGDLinearRegression()
            sgdlinear.fit(X_train_boston, y_train_boston, batch_size=batch, num_epochs=epoch_lim)
            pred_linear1 = sgdlinear.predict(X_test_boston)
            mse = sgdlinear.mean_squared_error(pred_linear1, y_test_boston)
            results[batch].append(mse)

            if batch not in results_log: results_log[batch] = []
            sgdlog = LogisticRegression()
            sgdlog.fit(X_train_wine, y_train_wine, batch_size=batch, num_epochs=epoch_lim)
            pred_log = sgdlog.predict(X_test_wine)
            accuracy = sgdlog.accuracy(y_test_wine, pred_log)
            results_log[batch].append(accuracy)
    for batch in minibatch_sizes:
        print(f'Linear results for batch size {batch}: {results[batch]}')
    #     pyplot.plot(epochs, results[batch], label=batch)
    # pyplot.legend(title="Minibatch Sizes")
    # pyplot.show()
    for batch in minibatch_sizes:
        print(f'Logistic results for batch size {batch}: {results_log[batch]}')
    #     pyplot.plot(epochs, results_log[batch], label=batch)
    # pyplot.legend(title="Minibatch Sizes")
    # pyplot.show()

    # 3.5
    print('---3.5: Different Learning Rates---')
    lr = [0.001, 0.01, 0.1]
    res_lin = []
    acc = []
    for r in lr:
        X1 = boston.drop('MEDV', axis=1)
        y1 = boston['MEDV']

        X2 = wine.data.features
        y2 = wine.data.targets

        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
        linear = SGDLinearRegression()
        linear.fit(X_train, y_train, learning_rate=r)
        pred_linear1 = linear.predict(X_test)
        mse = linear.mean_squared_error(pred_linear1, y_test)
        res_lin.append(mse)

        X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
        sgdlog = LogisticRegression()
        sgdlog.fit(X_train, y_train, learning_rate=r)
        pred_log = sgdlog.predict(X_test)
        accuracy = sgdlog.accuracy(y_test, pred_log)
        acc.append(accuracy)
    for i in range(len(res_lin)):
        print(f'Linreg MSE for lr={lr[i]}: {res_lin[i]}')
    for i in range(len(acc)):
        print(f'Logistic accuracy for lr={lr[i]}: {acc[i]}')
    # pyplot.plot(lr, res_lin)
    # pyplot.show()
    # pyplot.plot(lr, acc)
    # pyplot.show()

    # 3.6
    print('---3.6: Optimizing Metrics---')
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    epochs = [500, 750, 1000, 1500, 2000, 2500]
    max_lr = 0
    max_epoch = 0
    max_acc = 0
    for rate in learning_rates:
        for epoch in epochs:
            newlog = LogisticRegression()
            X2 = wine.data.features
            y2 = wine.data.targets
            X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
            newlog.fit(X_train, y_train, num_epochs=epoch, learning_rate=rate)
            pred_linear1 = newlog.predict(X_test)
            acc = newlog.accuracy(y_test, pred_linear1)
            if acc >= max_acc:
                max_acc = acc
                max_lr = rate
                max_epoch = epoch
    print(f'Maximum accuracy: {max_acc} using epochs: {max_epoch} and learning rate: {max_lr}')

    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    batch_size = [16, 32, 48, 64, 128, 256]
    max_lr = 0
    max_batch = 0
    max_mse = 99999
    for rate in learning_rates:
        for size in batch_size:
            newlin = SGDLinearRegression()
            X2 = boston.drop('MEDV', axis=1)
            y2 = boston['MEDV']
            X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
            newlin.fit(X_train, y_train, learning_rate=rate, batch_size=size)
            pred_linear1 = newlin.predict(X_test)
            mse = newlin.mean_squared_error(pred_linear1, y_test)
            if mse <= max_mse:
                max_mse = mse
                max_lr = rate
                max_batch = size
    print(f'Minimum MSE: {max_mse} using batch size: {max_batch} and learning rate: {max_lr}')

    #3.7
    print('---3.7: Gaussian Basis Functions---')
    newlinear = LinearRegression()
    X2 = boston.drop('MEDV', axis=1)
    y2 = boston['MEDV']
    X2 = newlinear.process(X2)
    for i in range(5):
        column_name = f'New_{i+1}'
        X2[column_name] = np.random.rand(len(X2))
    for index, x in X2.iterrows():
        for i in range(5):
            mu = X2.sample(n=1)
            new_feature = newlinear.gaussian_basis(mu, x)
            X2.at[index, f'New_{i+1}'] = new_feature
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    newlinear.fit(X_train, y_train)
    pred_linear1 = newlinear.predict(X_test)
    mse = newlinear.mean_squared_error(pred_linear1, y_test)
    print("With gaussian: " + str(mse))

    X3 = boston.drop('MEDV', axis=1)
    y3 = boston['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
    newlinear = LinearRegression()
    newlinear.fit(X_train, y_train)
    pred_linear1 = newlinear.predict(X_test)
    mse = newlinear.mean_squared_error(pred_linear1, y_test)
    print("Without gaussian: " + str(mse))

    # ADDITIONAL TESTS
    print('---ADDITIONAL TESTS---')
    print('---Momentum SGD---')
    momentums = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95]
    log_accs = []
    lin_mses = []
    for momentum in momentums:
        logistic = LogisticRegression()
        logistic.fit(X_train_wine, y_train_wine, momentum_rate=momentum)
        pred = logistic.predict(X_test_wine)
        log_accs.append(logistic.accuracy(y_test_wine, pred))

        linear = SGDLinearRegression()
        linear.fit(X_train_boston, y_train_boston, momentum_rate=momentum)
        pred = linear.predict(X_test_boston)
        lin_mses.append(linear.mean_squared_error(pred, y_test_boston))

    print(lin_mses)
    print(log_accs)
    figure, axis = pyplot.subplots(1, 2, layout='constrained')
    # axis[0].plot(momentums, lin_mses)
    # axis[0].set_title("Linear")
    # axis[1].plot(momentums, log_accs)
    # axis[1].set_title("Logistic")
    # pyplot.show()

    print('---Lp Regularization SGD---')
    regs = [0.5, 1, 1.5, 2, 2.5]
    log_accs = []
    lin_mses = []
    for reg in regs:
        logistic = LogisticRegression()
        logistic.fit(X_train_wine, y_train_wine, lambdaa=0.1, p=reg)
        pred = logistic.predict(X_test_wine)
        log_accs.append(logistic.accuracy(y_test_wine, pred))

        linear = SGDLinearRegression()
        linear.fit(X_train_boston, y_train_boston, lambdaa=0.1, p=reg)
        pred = linear.predict(X_test_boston)
        lin_mses.append(linear.mean_squared_error(pred, y_test_boston))

    print(log_accs)
    print(lin_mses)
    figure, axis = pyplot.subplots(1, 2, layout='constrained')
    # axis[0].plot(regs, lin_mses)
    # axis[0].set_title("Linear")
    # axis[1].plot(regs, log_accs)
    # axis[1].set_title("Logistic")
    # pyplot.show()


if __name__ == '__main__':
    main()