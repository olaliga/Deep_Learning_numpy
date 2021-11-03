# %%
import pickle
from sklearn import preprocessing
from numpy.lib.function_base import select
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# %%


class Model:
    def __init__(self):
        self.layers = []
        self.loss = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        # Forward pass
        for i, _ in enumerate(self.layers):
            forward = self.layers[i].forward(X)
            X = forward

        return forward

    def show_weight_final(self, i):
        return self.layers[i].show_weight()

    def show_latent(self, X):

        for i, _ in enumerate(self.layers):
            if i < len(self.layers)-3:
                forward = self.layers[i].forward(X)
                X = forward

        return forward

    def train(
        self,
        X_train,
        Y_train,
        learning_rate,
        epochs,
        loss_fcn,
        optimizer,
        batch_size,
        lamb=0,
        lamb2=0,
        debug=False,
        verbose=False
    ):
        n = X_train.shape[0]
        permutation = np.random.permutation(X_train.shape[0])
        for epoch in range(epochs):
            if n % batch_size > 0:
                times = int(n/batch_size)
            else:
                times = int(n/batch_size)
                times = times - 1

            # random index
            for time in range(times+1):
                if time != times:
                    index = permutation[batch_size*time:batch_size*(time+1)]
                    loss = self._run_epoch(
                        X_train[index].T,
                        Y_train[index],
                        learning_rate, loss_fcn, optimizer, epoch, lamb, lamb2,
                        debug)
                else:
                    index = permutation[batch_size*time:n]
                    loss = self._run_epoch(
                        X_train[index].T,
                        Y_train[index],
                        learning_rate, loss_fcn, optimizer, epoch, lamb, lamb2,
                        debug)

            self.loss.append(loss)

            if verbose:
                if epoch % 50 == 0:
                    print(f'Epoch: {epoch}. Loss: {loss}')

    def _run_epoch(self, X, Y, learning_rate, loss_fcn, optimizer, epoch, lamb, lamb2, debug=False):
        orig_data = X
        # Forward pass
        for i, _ in enumerate(self.layers):
            forward = self.layers[i].forward(input_val=X)
            X = forward
            if debug:
                print(f'forward {i} : {forward}')

        # Compute loss and first gradient
        if loss_fcn == 'binary':
            bce = BinaryCrossEntropy(forward, Y)
            gradient = bce.backward()
        else:
            mse = MeanSquaredError(forward, Y)
            gradient = mse.backward()

        if debug:
            print(f'gradient : {gradient}')
        # Backpropagation
        for i, _ in reversed(list(enumerate(self.layers))):

            if self.layers[i].type != 'Linear':
                gradient = self.layers[i].backward(gradient)
            else:
                gradient, dW, dB = self.layers[i].backward(gradient)
                self.layers[i].optimize(
                    dW, dB, learning_rate, optimizer, epoch, lamb, lamb2)

        if loss_fcn == 'binary':
            bce = BinaryCrossEntropy(self.predict(orig_data), Y)
            error_output = bce.forward()
        else:
            mse = MeanSquaredError(self.predict(orig_data), Y)
            error_output = mse.forward()

        return error_output


class Layer:
    """Layer abstract class"""

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def optimize(self):
        pass


class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(output_dim, 1)
        self.type = 'Linear'
        self.dW_last = []
        self.dB_last = []
        self.rw_last = []
        self.rb_last = []

    def show_weight(self):
        return self.weights, self.biases

    def __str__(self):
        output_weight = self.weights
        # return f"{self.type} Layer"
        return output_weight

    def forward(self, input_val):
        self._prev_acti = input_val
        return np.matmul(self.weights, input_val) + self.biases

    def backward(self, dA):
        dW = np.dot(dA, self._prev_acti.T)
        dB = dA.sum(axis=1, keepdims=True)

        delta = np.dot(self.weights.T, dA)

        return delta, dW, dB

    def optimize(self, dW, dB, rate, optimizer, epoch, lamb, lamb2):
        if optimizer == 'sgd':
            self.weights = self.weights - rate * dW
            self.biases = self.biases - rate * dB
        elif optimizer == 'momentum':
            if epoch == 0:
                self.dW_last = np.zeros(dW.shape)
                self.dB_last = np.zeros(dB.shape)

            dW_final = lamb*self.dW_last - (1-lamb)*dW
            dB_final = lamb*self.dB_last - (1-lamb)*dB
            self.weights = self.weights + rate * dW_final
            self.biases = self.biases + rate * dB_final
            self.dW_last = dW_final
            self.dB_last = dB_final
        elif optimizer == 'Adam':
            if epoch == 0:
                self.dW_last = np.zeros(dW.shape)
                self.dB_last = np.zeros(dB.shape)
                self.rw_last = np.zeros(dW.shape)
                self.rb_last = np.zeros(dB.shape)

            dW_final = lamb*self.dW_last - (1-lamb)*dW
            dB_final = lamb*self.dB_last - (1-lamb)*dB

            rw_final = lamb2*self.rw_last + (1-lamb2)*dW**2
            rb_final = lamb2*self.rb_last + (1-lamb2)*dB**2

            self.weights = self.weights + rate/rw_final * dW_final
            self.biases = self.biases + rate/rb_final * dB_final

            self.dW_last = dW_final
            self.dB_last = dB_final
            self.rb_last = rb_final
            self.rw_last = rw_final


class ReLU(Layer):
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'ReLU'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = np.maximum(0, input_val)
        return self._prev_acti

    def backward(self, dJ):
        return dJ * np.heaviside(self._prev_acti, 0)


class LeakyReLU(Layer):
    def __init__(self, output_dim, slope):
        self.units = output_dim
        self.type = 'LeakyReLU'
        self.slope = slope

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = np.maximum(
            0, input_val) + self.slope*np.minimum(0, input_val)
        return self._prev_acti

    def backward(self, dJ):
        return dJ * (np.heaviside(self._prev_acti, 0) + self.slope*np.heaviside(-self._prev_acti, 0))


class Sigmoid(Layer):
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'Sigmoid'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = 1.0 / (1.0 + np.exp(-input_val))
        return self._prev_acti

    def backward(self, dJ):
        sig = self._prev_acti
        return dJ * sig * (1.0 - sig)


class Linear_output(Layer):
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'Linear_output'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = input_val
        return self._prev_acti

    def backward(self, dJ):
        sig = self._prev_acti
        return dJ


class MeanSquaredError(Layer):
    def __init__(self, predicted, real):
        self.predicted = predicted
        self.real = real
        self.type = 'Mean Squared Error'

    def forward(self):
        n = len(self.real)
        return np.power(self.predicted - self.real, 2).mean()

    def backward(self):
        n = len(self.real)
        return 2 * (self.predicted - self.real)/n


class BinaryCrossEntropy(Layer):
    def __init__(self, predicted, real):
        self.real = real
        self.predicted = predicted
        self.type = 'Binary Cross-Entropy'

    def forward(self):
        n = len(self.real)
        loss = np.nansum(-self.real * np.log(self.predicted) -
                         (1 - self.real) * np.log(1 - self.predicted)) / n

        return np.squeeze(loss)

    def backward(self):
        n = len(self.real)
        return (-(self.real / self.predicted) + ((1 - self.real) / (1 - self.predicted))) / n


def generate_data(samples, shape_type='circles', noise=0.05):
    # We import in the method for the shake of simplicity
    import matplotlib
    import pandas as pd

    from matplotlib import pyplot as plt
    from sklearn.datasets import make_moons, make_circles
    if shape_type is 'moons':
        X, Y = make_moons(n_samples=samples, noise=noise)
    elif shape_type is 'circles':
        X, Y = make_circles(n_samples=samples, noise=noise)
    else:
        raise ValueError(
            f"The introduced shape {shape_type} is not valid. Please use 'moons' or 'circles' ")

    data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))

    return data


def plot_generated_data(data):
    ax = data.plot.scatter(x='x', y='y', figsize=(16, 12), color=data['label'],
                           cmap=matplotlib.colors.ListedColormap(['skyblue', 'salmon']), grid=True)

    return ax


def accuracy(pred, label):
    return(1 - np.mean((np.abs(np.array(pred - label)))))


################################################
########## problem 2 classification ############
################################################

# %% 1 = g, 0 = b
data_classification = pd.read_csv('ionosphere_data.csv')
del data_classification['var2']
Y_classification = np.array((data_classification['target'] == 'g')).astype(int)
X_classification = np.array(data_classification.iloc[:, 0:33])

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification, Y_classification, test_size=0.2, random_state=1023)

# %%
'''
# Create model
model_classification = Model()

# Add layers
model_classification.add(Linear(33, 16))
model_classification.add(LeakyReLU(16, 0))

model_classification.add(Linear(16, 32))
model_classification.add(Sigmoid(32))

model_classification.add(Linear(32, 1))
model_classification.add(Sigmoid(1))
'''
# %%
# Create model
model_classification = Model()

# Add layers
model_classification.add(Linear(33, 3))
model_classification.add(ReLU(3))

model_classification.add(Linear(3, 3))
model_classification.add(ReLU(3))

model_classification.add(Linear(3, 1))
model_classification.add(Sigmoid(1))


# %%
# epoch 10
# Train model
model_classification.train(X_train=X_train_class,
                           Y_train=y_train_class,
                           learning_rate=5*10**-3,
                           epochs=10,
                           loss_fcn='binary',
                           optimizer='sgd',
                           batch_size=64,
                           lamb=0.3,
                           lamb2=0.45,
                           debug=False,
                           verbose=True)
latent3_10 = model_classification.show_latent(X_test_class.T).T
# %%
# plot


def plot_latent_3d(latent, index_1, index_0, title, angle=(200, 50)):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(latent[np.ix_(index_1, [0])], latent[np.ix_(
        index_1, [1])], latent[np.ix_(index_1, [2])], cmap='red', marker='^', label='g')

    ax.scatter(latent[np.ix_(index_0, [0])], latent[np.ix_(index_0, [
               1])], latent[np.ix_(index_0, [2])], cmap='blue', marker='^', label='b')

    ax.set_xlabel('latent 1')
    ax.set_ylabel('latent 2')
    ax.set_zlabel('latent 3')
    ax.azim, ax.elev = angle
    # 顯示圖例
    ax.legend()
    plt.title(title)
    # 顯示圖形
    plt.show()


def plot_latent_2d(latent, index_1, index_0, title):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(latent[np.ix_(index_1, [0])], latent[np.ix_(
        index_1, [1])], color='red', marker='o', label='g')

    ax.scatter(latent[np.ix_(index_0, [0])], latent[np.ix_(
        index_0, [1])], color='blue', marker='o', label='b')

    ax.set_xlabel('latent 1')
    ax.set_ylabel('latent 2')
    # 顯示圖例
    ax.legend()
    plt.title(title)
    # 顯示圖形
    plt.show()


# %%
index_1 = [i for ele in np.where(y_test_class == 1) for i in ele.flatten()]
index_0 = [i for ele in np.where(y_test_class == 0) for i in ele.flatten()]

plot_latent_3d(latent3_10, index_1, index_0, '3D feature epoch 10')


# %%
# Train model
model_classification.train(X_train=X_train_class,
                           Y_train=y_train_class,
                           learning_rate=5*10**-3,
                           epochs=4990,
                           loss_fcn='binary',
                           optimizer='sgd',
                           batch_size=64,
                           lamb=0.3,
                           lamb2=0.45,
                           debug=False,
                           verbose=True)

latent3_10000 = model_classification.show_latent(X_test_class.T).T
# %%
# plot
plot_latent_3d(latent3_10000, index_1, index_0, '3D feature epoch 5000')

# %%
plt.plot(model_classification.loss)
plt.title("Learning Curve for Classification Problem")
plt.ylabel("CrossEntropy")
plt.xlabel("epoch")

# %%

prediction_class_train = model_classification.predict(X_train_class.T)
prediction_class_train_label = np.array(
    prediction_class_train > 0.5).astype(int).flatten()

cross_entropy_train = np.mean(-y_train_class * np.log(prediction_class_train) -
                              (1 - y_train_class) * np.log(1 - prediction_class_train))

prediction_class = model_classification.predict(X_test_class.T)
prediction_class_label = np.array(prediction_class > 0.5).astype(int).flatten()
cross_entropy_test = np.mean(-y_test_class * np.log(prediction_class) -
                             (1 - y_test_class) * np.log(1 - prediction_class))

# %%
accuracy(prediction_class_label, y_test_class)

# %%
# Create model
model_classification_2 = Model()

# Add layers
model_classification_2.add(Linear(33, 3))
model_classification_2.add(ReLU(3))

model_classification_2.add(Linear(3, 2))
model_classification_2.add(ReLU(2))

model_classification_2.add(Linear(2, 1))
model_classification_2.add(Sigmoid(1))

# %%
# Train model
model_classification_2.train(X_train=X_train_class,
                             Y_train=y_train_class,
                             learning_rate=5*10**-3,
                             epochs=10,
                             loss_fcn='binary',
                             optimizer='sgd',
                             batch_size=64,
                             lamb=0.3,
                             lamb2=0.45,
                             debug=False,
                             verbose=True)

latent3_10_2 = model_classification_2.show_latent(X_test_class.T).T
plot_latent_2d(latent3_10_2, index_1, index_0, '2D feature epoch 10')

# %%
# Train model
model_classification_2.train(X_train=X_train_class,
                             Y_train=y_train_class,
                             learning_rate=5*10**-3,
                             epochs=4990,
                             loss_fcn='binary',
                             optimizer='sgd',
                             batch_size=64,
                             lamb=0.3,
                             lamb2=0.45,
                             debug=False,
                             verbose=True)

latent3_10000_2 = model_classification_2.show_latent(X_test_class.T).T
plot_latent_2d(latent3_10000_2, index_1, index_0, '2D feature epoch 5000')


#############################################################
###################  problem 1 regression ##################
#############################################################
# %%
data_regression = pd.read_csv('energy_efficiency_data.csv')
data_regression = pd.concat([data_regression, pd.get_dummies(
    data_regression.Orientation, drop_first=True)], axis=1)
del data_regression['Orientation']
data_regression.rename({3.0: 'Orientation.3',
                        4.0: 'Orientation.4',
                        5.0: 'Orientation.5'}, axis=1, inplace=True)


data_regression.rename(
    {'Glazing Area Distribution': 'GAD'}, axis=1, inplace=True)
data_regression = pd.concat([data_regression, pd.get_dummies(
    data_regression.GAD, drop_first=True)], axis=1)
del data_regression['GAD']
data_regression.rename({1.0: 'GAD.1',
                        2.0: 'GAD.2',
                        3.0: 'GAD.3',
                        4.0: 'GAD.4',
                        5.0: 'GAD.5'}, axis=1, inplace=True)

Y_regression = np.array(data_regression['Heating Load'])
X_regression = np.array(data_regression.drop(columns=['Heating Load']))
#X_regression = np.array(data_regression[['Cooling Load', 'Surface Area']])

# %%
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression, Y_regression, test_size=0.25, random_state=1023)

# %%
min_max_scaler = preprocessing.MinMaxScaler()
sd_scaler = preprocessing.StandardScaler()
X_train_reg = sd_scaler.fit_transform(X_train_reg)
X_test_reg = sd_scaler.transform(X_test_reg)

# %%
# Create model
model_regression = Model()

# Add layers
model_regression.add(Linear(X_regression.shape[1], 8))
model_regression.add(LeakyReLU(8, 0.01))

model_regression.add(Linear(8, 32))
model_regression.add(LeakyReLU(32, 0.2))

model_regression.add(Linear(32, 16))
model_regression.add(LeakyReLU(16, 0.01))

model_regression.add(Linear(16, 1))
model_regression.add(Linear_output(1))


# %%
# Train model
model_regression.train(X_train=X_train_reg,
                       Y_train=y_train_reg,
                       learning_rate=1*10**-5,
                       epochs=15000,
                       loss_fcn='linear',
                       optimizer='sgd',
                       batch_size=64,
                       lamb=0.3,
                       lamb2=0.45,
                       verbose=True)

# %%
plt.plot(model_regression.loss)
plt.title("Learning Curve for regression Problem")
plt.ylabel("MSE")
plt.xlabel("epoch")

# %%
plt.plot(model_regression.loss)
plt.ylim(0, 30)
plt.title("Learning Curve for regression Problem")
plt.ylabel("MSE")
plt.xlabel("epoch")

# %%
train_result = model_regression.predict(X_train_reg.T).T
rmse_train = np.mean((train_result.flatten() - y_train_reg)**2)**0.5

# %%
plt.plot(y_train_reg,  train_result.flatten(), 'o', color='black')
plt.xlabel("True label")
plt.ylabel("Prediction")
plt.title('Prediction for training data')

# %%
prediction_regression = model_regression.predict(X_test_reg.T).T
rmse_test = np.mean((prediction_regression.flatten() - y_test_reg)**2)**0.5


# %%
plt.plot(y_test_reg,  prediction_regression.flatten(), 'o', color='black')
plt.xlabel("True label")
plt.ylabel("Prediction")
plt.title('Prediction for testing data')

# %%
# feature selection
threshold = 0.5
rmse_new = 0
no_variable = 9
# %%
select_index_total = [[i] for i in range(7)]
select_index_total.append([i+7 for i in range(3)])
select_index_total.append([i+10 for i in range(5)])

remove_index = []
orig_index = [dum for dum in range(X_train_reg.shape[0])]

loss_featue_selection_final = []

while rmse_new < rmse_train + threshold and len(remove_index) < no_variable - 1:
    select_index_dum = [i for i in select_index_total if i not in remove_index]
    loss_featue_selection = []

    for select_remove in select_index_dum:
        select_index = [i for ele in select_index_dum if ele not in [
            select_remove] for i in ele]
        feature_selection_data = X_train_reg[np.ix_(orig_index, select_index)]

        # model
        model_regression_cv = Model()

        model_regression_cv.add(Linear(feature_selection_data.shape[1], 8))
        model_regression_cv.add(LeakyReLU(8, 0.01))

        model_regression_cv.add(Linear(8, 32))
        model_regression_cv.add(LeakyReLU(32, 0.2))

        model_regression_cv.add(Linear(32, 16))
        model_regression_cv.add(LeakyReLU(16, 0.01))

        model_regression_cv.add(Linear(16, 1))
        model_regression_cv.add(Linear_output(1))

        model_regression_cv.train(X_train=feature_selection_data,
                                  Y_train=y_train_reg,
                                  learning_rate=1*10**-5,
                                  epochs=15000,
                                  loss_fcn='linear',
                                  optimizer='sgd',
                                  batch_size=64,
                                  lamb=0.3,
                                  lamb2=0.45,
                                  verbose=False)

        prediction_regression = model_regression_cv.predict(
            feature_selection_data.T).T
        loss = np.mean((prediction_regression.flatten() - y_train_reg)**2)**0.5
        print(f'Variable: {select_index}. Loss: {loss}')

        loss_featue_selection.append((select_index, loss))

    loss_featue_selection_final += loss_featue_selection

    save_index_list = []
    for i in min(loss_featue_selection, key=lambda t: t[1])[0]:
        if i in [7, 8, 9]:
            save_index_list.append([7, 8, 9])
        elif i in [10, 11, 12, 13, 14]:
            save_index_list.append([10, 11, 12, 13, 14])
        else:
            save_index_list.append([i])

    save_index_list = [list(x) for x in set(tuple(x) for x in save_index_list)]

    for ele in select_index_total:
        if ele not in save_index_list:
            if ele not in remove_index:
                remove_index.append(ele)

    rmse_new = min(loss_featue_selection, key=lambda t: t[1])[1]

print("Done!")


# %%
feature_selection_data_final = X_train_reg[:, [6, 7, 8, 9]]

# model
model_regression_final = Model()

model_regression_final.add(Linear(feature_selection_data_final.shape[1], 8))
model_regression_final.add(LeakyReLU(8, 0.01))

model_regression_final.add(Linear(8, 32))
model_regression_final.add(LeakyReLU(32, 0.2))

model_regression_final.add(Linear(32, 16))
model_regression_final.add(LeakyReLU(16, 0.01))

model_regression_final.add(Linear(16, 1))
model_regression_final.add(Linear_output(1))

model_regression_final.train(X_train=feature_selection_data_final,
                             Y_train=y_train_reg,
                             learning_rate=1*10**-5,
                             epochs=15000,
                             loss_fcn='linear',
                             optimizer='sgd',
                             batch_size=128,
                             lamb=0.3,
                             lamb2=0.45,
                             verbose=False)

# %%
prediction_regression_final = model_regression_final.predict(
    X_test_reg[:, [6, 7, 8, 9]].T).T

loss_final = np.mean(
    (prediction_regression_final.flatten() - y_test_reg)**2)**0.5


# %%
# saving feature selection data

with open("loss_featue_selection.txt", "wb") as fp:
    pickle.dump(loss_featue_selection_final, fp)

# %%
feature_plot_x = [9]
feature_plot_x = feature_plot_x + \
    [j for j in [8, 7, 6, 5, 4, 3, 2] for i in range(j+1)]
feature_plot_y = [rmse_train]
feature_plot_y = feature_plot_y + [ele[1]
                                   for ele in loss_featue_selection_final]

plt.scatter(feature_plot_x, feature_plot_y)
plt.title("Feature Selection Plot")
plt.xlabel("Number of features selected")
plt.ylabel("Training data rmse")
plt.axhline(y=rmse_train + threshold, color='r', linestyle='--',
            lw=2, label='Threshold')
plt.annotate('6 7 8 9', (2+0.2, 1.95))
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.show()


# %%
