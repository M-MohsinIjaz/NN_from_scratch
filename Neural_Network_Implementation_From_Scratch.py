import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import image as img
import glob
from sklearn.manifold import TSNE
import pickle
from sklearn.metrics import confusion_matrix


def loadDataset(path):
    print('Loading Dataset...')
    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(10):
        for filename in glob.glob(path + '//train//' + str(i) + '//*.png'):
            im = img.imread(filename)
            train_x.append(im)
            train_y.append(i)
    for i in range(10):
        for filename in glob.glob(path + '//test//' + str(i) + '//*.png'):
            im = img.imread(filename)
            test_x.append(im)
            test_y.append(i)
    print('Dataset loaded...')
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


def meanSubtraction(images):
    # compute the mean image
    mean_image = np.mean(images, axis=0)  # shape: (784,)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    axes[0].imshow(images[0].reshape(28, 28), cmap='gray')
    axes[0].set_title('Orignal image')
    # Subtract mean image
    for image in images:
        image = image - mean_image

    # Plot firts image of data set as sample
    axes[1].imshow(mean_image.reshape(28, 28), cmap='gray')
    axes[1].set_title('mean image')
    axes[2].imshow(images[0].reshape(28, 28), cmap='gray')
    axes[2].set_title('mean subtracted image')
    plt.show()
    return images


def get_one_hot_encoding(label, num_classes):
    onehot_encoded = np.zeros((label.shape[0], num_classes))

    # Loop through each sample in the input array and set the corresponding
    # value in the one-hot encoded array to 1
    for i in range(label.shape[0]):
        class_index = label[i][0]
        onehot_encoded[i][class_index] = 1
    return onehot_encoded


def get_splited_datasets(path):
    X_train, Y_train, X_test, Y_test = loadDataset(path)

    # Reshaping data
    X_train = X_train.reshape(X_train.shape[0], 1, -1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1, -1)
    Y_train = Y_train.reshape(Y_train.shape[0], -1)
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_TRAIN = Y_train
    Y_train = get_one_hot_encoding(Y_train, 10)
    X_test = X_test.reshape(X_test.shape[0], 1, -1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1, -1)
    Y_test = Y_test.reshape(Y_test.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_TEST = Y_test
    Y_test = get_one_hot_encoding(Y_test, 10)
    # get a random permutation of indices
    perm = np.random.permutation(X_train.shape[0])

    # shuffle both arrays using the permutation
    X_shuffled = X_train[perm]
    Y_shuffled = Y_train[perm]
    Y_TRAIN = Y_TRAIN[perm]
    X_train = X_shuffled.T
    Y_train = Y_shuffled.T
    X_test = X_test.T
    Y_test = Y_test.T
    print("shape of X_train :", X_train.shape)
    print("shape of Y_train :", Y_train.shape)
    print("shape of X_test :", X_test.shape)
    print("shape of Y_test :", Y_test.shape)
    return X_train, Y_train, X_test, Y_test, Y_TRAIN, Y_TEST


def show_predicted_actual(X, Y, index):
    print('Predicted Values is {} \r\n'.format(Y[index]))
    print('Actual Image \r\n')
    plt.imshow(X[:, index].reshape(28, 28), cmap='gray')
    plt.show()


def show_random_img_from_dataset(X):
    index = random.randrange(0, X.shape[1])
    plt.imshow(X[:, index].reshape(28, 28), cmap='gray')
    plt.show()


def derivative_tanh(x):
    return (1 - np.power(np.tanh(x), 2))


def derivative_relu(x):
    return np.array(x > 0, dtype=np.float32)


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    expX = np.exp(x)
    return expX / np.sum(expX, axis=0)


def sigmoid(s):
    # activation function
    return 1 / (1 + np.exp(-s))  # apply sigmoid function on s and return it's value


def sigmoid_derivative(s):
    # derivative of sigmoid
    return sigmoid(s) * (1 - sigmoid(s))  # apply derivative of sigmoid on s and return it's value


def initialize_parameters(input_dim, neurons_per_layer, no_of_layers):
    input = np.array([input_dim])
    layers_per_layer = np.concatenate((input, neurons_per_layer))
    L = len(layers_per_layer)
    parameters = {
        "w": [],
        "b": []
    }
    forward_cache = {
        "z": [None] * (L - 1),
        "a": [None] * (L - 1)
    }
    gradients = {
        "dw": [None] * (L - 1),
        "db": [None] * (L - 1),
        "dz": [None] * (L - 1)
    }
    for i in range(0, L - 1):
        parameters['w'].append(np.random.randn(layers_per_layer[i + 1], layers_per_layer[i]) * 0.1)

        parameters['b'].append(np.zeros((layers_per_layer[i + 1], 1)))
    for i in range(0, L - 1):
        print(i, parameters['w'][i].shape)
        print(i, parameters['b'][i].shape)
    return parameters, forward_cache, gradients


def forward_propagation(x, parameters, forward_cache):
    for i in range(0, len(parameters['w'])):
        if i == 0:
            forward_cache['z'][i] = np.dot(parameters['w'][i], x) + parameters['b'][i]
            forward_cache['a'][i] = sigmoid(forward_cache['z'][i])
        elif i == (len(parameters['w']) - 1):
            forward_cache['z'][i] = np.dot(parameters['w'][i], forward_cache['a'][i - 1]) + parameters['b'][i]
            forward_cache['a'][i] = softmax(forward_cache['z'][i])
        else:
            forward_cache['z'][i] = np.dot(parameters['w'][i], forward_cache['a'][i - 1]) + parameters['b'][i]
            forward_cache['a'][i] = sigmoid(forward_cache['z'][i])
    return forward_cache


def cost_function(a2, y):
    m = y.shape[1]

    cost = -(1 / m) * np.sum(y * np.log(a2))

    return cost


def backward_prop(x, y, parameters, forward_cache, gradients):
    m = x.shape[1]
    gradients['dz'][-1] = (forward_cache['a'][-1] - y)
    gradients['dw'][-1] = (1 / m) * np.dot(gradients['dz'][-1], forward_cache['a'][-2].T)
    gradients['db'][-1] = (1 / m) * np.sum(gradients['dz'][-1], axis=1, keepdims=True)
    pms = len(parameters['w'])
    for i in range(pms - 2, -1, -1):
        gradients['dz'][i] = (1 / m) * np.dot(parameters['w'][i + 1].T, gradients['dz'][i + 1]) * sigmoid_derivative(
            forward_cache['a'][i])
        if i == 0:
            gradients['dw'][i] = (1 / m) * np.dot(gradients['dz'][i], x.T)
        else:
            gradients['dw'][i] = (1 / m) * np.dot(gradients['dz'][i], forward_cache['a'][i - 1].T)

        gradients['db'][i] = (1 / m) * np.sum(gradients['dz'][i], axis=1, keepdims=True)
    return gradients


def update_parameters(parameters, gradients, learning_rate):
    for i in range(0, len(parameters['w'])):
        parameters['w'][i] = parameters['w'][i] - learning_rate * gradients['dw'][i]
        parameters['b'][i] = parameters['b'][i] - learning_rate * gradients['db'][i]

    return parameters


def model(input_dim, neurons_per_layer, no_of_layers, training=True, model_name='mymodel'):
    cost_list = []

    # intialize parameters
    parameters, forward_cache, gradients = initialize_parameters(input_dim, neurons_per_layer, no_of_layers)

    if training is True:
        return parameters, forward_cache, gradients
    else:
        # Load model params
        parameters = loadModel(model_name)

    # = initialize_parameters(n_x, n_h, n_y)
    return parameters, forward_cache, gradients


def train(epochs, x, y, parameters, forward_cache, gradients, learning_rate):
    cost_list = []
    accuracy_list = []
    # Dataset is very large so coverting it into batches
    batch_size = 100
    X = np.split(x, x.shape[1] / batch_size, axis=1)
    Y = np.split(y, y.shape[1] / batch_size, axis=1)

    number_of_batches = len(X)
    for i in range(epochs):
        t_cost = 0
        t_accu = 0
        for x, y in zip(X, Y):
            forward_cache = forward_propagation(x, parameters, forward_cache)
            cost = cost_function(forward_cache['a'][-1], y)
            acc, acc_list = accuracy(forward_cache['a'][-1].T, y.T)
            gradients = backward_prop(x, y, parameters, forward_cache, gradients)

            parameters = update_parameters(parameters, gradients, learning_rate)
            t_cost += cost
            t_accu += acc

        cost_list.append(t_cost / number_of_batches)
        accuracy_list.append(t_accu / number_of_batches)

        if (i % (epochs / 10) == 0):
            print("Cost after", i, "epochs is :", t_cost / number_of_batches)
            print("ACCuracy after", i, "epochs is :", t_accu / number_of_batches)

    return parameters, cost_list, accuracy_list


def predict(x, parameters, forward_cache):
    prediction_num = np.zeros((x.shape[1], 1))
    prediction = forward_propagation(x, parameters, forward_cache)
    prediction = prediction['a'][-1].T
    for sample in range(0, prediction.shape[0]):
        prediction_num[sample] = prediction[sample].argmax()
    return prediction_num


def accuracy(pred_values, target_values):
    matches = 0.0
    matched_sample_indexes = []
    for sample in range(0, target_values.shape[0]):
        if pred_values[sample].argmax() == target_values[sample].argmax():
            matched_sample_indexes.append(sample)
        else:
            pass
    return len(matched_sample_indexes) / target_values.shape[0], matched_sample_indexes


def plot_loss_acc(epochs, cost_list, accuracy_list):
    t = np.arange(0, epochs)
    plt.plot(t, cost_list, label='Training Loss')
    plt.plot(t, accuracy_list, label='Accuracy')
    plt.xlabel('Number of Epoches')
    plt.ylabel('Training Loss and Accuracy')
    plt.legend()
    plt.show()


def saveModel(name, model):
    # save your trained model, it is your interpretation how, which and what data you store
    # which you will use later for prediction
    # save the trained model as a .mdl file
    with open('{}.mdl'.format(name), 'wb') as f:
        pickle.dump(model, f)


def loadModel(name):
    # load your trained model, load exactly how you stored it.
    with open('{}.mdl'.format(name), 'rb') as f:
        model = pickle.load(f)
    return dict(model)


def visualize_data_tsne(dataset_x, dataset_y):
    dataset_x = dataset_x.reshape(dataset_x.shape[0], -1)
    # Define T-sne model and fit it to the features
    # Note this step can be removed as our data is already in 2D
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(dataset_x)

    # create a figure
    plt.figure(figsize=(10, 10))
    # plot data points
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dataset_y, cmap="jet")
    # add colorbar
    plt.colorbar()
    plt.legend()
    plt.show()


def plot_confusion_matrix(Y, predicted_values):
    # 0 to 9 class labels
    class_labels = np.arange(10)

    # Calculate the confusion matrix
    cm = confusion_matrix(Y, predicted_values, labels=class_labels)
    # Create figure of size 10x10
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm)
    # Label axis
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_labels, yticklabels=class_labels,
           ylabel='Actual', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), ha="right")
    # Loop over cm matrix and actual values as text in the boxes.
    fmt = 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), color="white")

    fig.tight_layout()
    plt.show()


def meanSubtraction(images):
    # compute the mean image
    mean_image = np.mean(images, axis=0)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    axes[0].imshow(images[0].reshape(28, 28), cmap='gray')
    axes[0].set_title('Orignal image')
    # Subtract mean image
    for image in images:
        image = image - mean_image

    # Plot firts image of data set as sample
    axes[1].imshow(mean_image.reshape(28, 28), cmap='gray')
    axes[1].set_title('mean image')
    axes[2].imshow(images[0].reshape(28, 28), cmap='gray')
    axes[2].set_title('mean subtracted image')
    plt.show()
    return images


def prediction_verbose(predicted_values, Y):
    count = 0
    print(len(predicted_values))
    print(Y.shape)
    print(predicted_values.shape)
    for sample in range(0, len(predicted_values) - 1):
        if predicted_values[sample] == Y[sample].argmax():
            count += 1
            print('Predicted {} , Target {}'.format(predicted_values[sample], Y[sample].argmax()))
    print("Total Correct = {}".format(count))
    print("Total Wrong = {}".format(Y.shape[0] - count))


def main_method():
    path = r'Task3_MNIST_Data'
    # Y_TRAIN,Y_TEST are orignally loaded datasets which are used fot T-SNE
    X_train, Y_train, X_test, Y_test, Y_TRAIN, Y_TEST = get_splited_datasets(path)
    no_of_layers = 3
    neurons_per_layer = [128, 64, 10]
    input_dim = X_train.shape[0]
    learning_rate = 0.1
    epochs = 2000
    training = False
    testing = True
    verbose = False
    t_sne_visualize = True
    pre_processing = True
    if pre_processing is True:
        # Data Pre-Processing
        X_train = meanSubtraction(X_train.T)
        X_test = meanSubtraction(X_test.T)

        # Reverting shapes back
        X_train = X_train.T
        X_test = X_test.T

    if t_sne_visualize is True:
        visualize_data_tsne(X_train[:, :20000].T, Y_TRAIN.T[:, :20000])

    # Initalize parameters
    parameters, forward_cache, gradients = model(input_dim, neurons_per_layer, no_of_layers, training=training)

    if training is True:
        # Train NN
        parameters, cost_list, accu = train(epochs, X_train, Y_train, parameters, forward_cache, gradients,
                                            learning_rate)

        if t_sne_visualize is True:
            for i in range(0, len(forward_cache['a']) - 1):
                print('T-SNE for hidden Layer {}'.format(i + 1))
                number_of_samples = int(forward_cache['a'][i].shape[1])
                visualize_data_tsne(forward_cache['a'][i][:, :number_of_samples].T, Y_TRAIN.T[:, :number_of_samples])
        # Plot Training Loss and Accuracy
        plot_loss_acc(epochs, cost_list, accu)
        saveModel('mymodel', parameters)

        # Predict values on training Dataset
        predicted_values = predict(X_train, parameters, forward_cache)
        plot_confusion_matrix(Y_TRAIN, predicted_values)
    if testing is True:

        ##################################################
        #                   Testing                      #
        #                                                #
        ##################################################
        # Predict values on testing Dataset
        predicted_values = predict(X_test, parameters, forward_cache)
        plot_confusion_matrix(Y_TEST, predicted_values)

        # Show some predicted values
        for i in range(0, 5):
            random_sample = random.randrange(i, X_test.shape[1])
            show_predicted_actual(X_test, predicted_values, random_sample)
        if verbose is True:
            prediction_verbose(predicted_values, Y_test)
        # Check testing accuracy
        pred_values = forward_propagation(X_test, parameters, forward_cache)
        print('Testing Accuracy is ={}'.format(accuracy(pred_values['a'][-1].T, Y_test.T)[0]))
        if t_sne_visualize is True:
            for i in range(0, len(forward_cache['a']) - 1):
                print('T-SNE for hidden Layer {}'.format(i + 1))
                number_of_samples = int(forward_cache['a'][i].shape[1])
                visualize_data_tsne(forward_cache['a'][i][:, :number_of_samples].T, Y_TEST.T[:, :number_of_samples])


if __name__ == '__main__':
    main_method()
