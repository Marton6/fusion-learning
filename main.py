import pandas
import numpy as np
import tensorflow as tf
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from scipy import stats
import matplotlib.pyplot as plt

epochs = 100

raw_dataset = pandas.read_excel('dataset.xls', header=0).to_numpy()
column_labels = raw_dataset[0, 1:-1]
dataset = raw_dataset[1:, 1:].astype(np.float64)
np.random.shuffle(dataset)
split_index = int(len(dataset)*.8)
training_set, test_set = dataset[:split_index], dataset[split_index:]

def negloglik(y, distr):
    return -distr.log_prob(y)

distributions = ['norm', 'exponweib', 'weibull_max', 'weibull_min', 'chi2', 'pareto', 'lognorm', 'beta', 'chi', 'pearson3', 'genextreme', 'expon', 'cauchy', 'cosine', 'powerlaw', 'gamma', 'logistic', 'lomax', 'maxwell', 'rdist', 'uniform', 'vonmises', 'wald', 'wrapcauchy']

def find_feature_dist(index, data):
    # TODO implement ks test to find best matching distribution 
    
    best_p = 0
    best_dist_name = ''
    best_dist_params = ''
    for dist_name in distributions:    
        dist = getattr(stats, dist_name)
        datac = np.copy(data)
        params = dist.fit(datac)
        _, p = stats.kstest(datac, dist_name, args=params)
        if p > best_p:
            best_p = p
            best_dist_name = dist_name
            best_dist_params = params

    print("Found best_dist="+best_dist_name+" for feature=" + column_labels[index])
    return best_dist_name, best_dist_params
    
    '''
    # For now we use the best distributions claimed in the paper and see if we get the same results
    dist = best_dist
    if column_labels[index] == 'AGE':
        dist = getattr(stats, 'beta')
        params = dist.fit(data, floc=0, fscale=100)
        dist = 'beta'
    elif column_labels[index] == 'LIMIT_BAL':
        dist = getattr(stats, 'gamma')
        params = dist.fit(data)
        dist = 'gamma'
    else:
        dist = getattr(stats, 'norm')
        params = dist.fit(data)
        dist = 'norm'
    
    return dist, params'''

def client(data):
    distributions = []
    # last column is the label, so we don't need to find its distribution
    for col in range(len(data[0])-1):
        dist_name, params = find_feature_dist(col, data[:,col])
        distributions.append({
            'name': dist_name,
            'params': params,
        })

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(23,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
    x, y = data[:, :-1], data[:, -1]
    #print(Counter(y))
    oversample = RandomOverSampler(sampling_strategy='minority')
    x_o, y_o = oversample.fit_resample(x, y)
    #print(Counter(y_o))
#    train_dataset = tf.data.Dataset.from_tensor_slices((data[:, :-1], data[:, -1]))
#    train_dataset = train_dataset.batch(32, drop_remainder=True)
    model.fit(x_o, y_o, batch_size=32, epochs=epochs, verbose=0)
    y_pred = model.predict(x, batch_size=32, verbose=0)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(y, y_pred_bool))
    # _, test_acc = model.evaluate(data[:,:-1], data[:,-1])
    # print("Test acc "+ str(test_acc))
    return model, distributions

# Clients do computations
models = []
distribution_lists = []
for i in range(10):
    left_index = int(i*len(training_set)/10)
    right_index = int((i+1)*len(training_set)/10)
    client_dataset = training_set[left_index:right_index]
    model, distribution = client(client_dataset)
    models.append(model)
    distribution_lists.append(distribution)
    print(i+1)

    # Check distributions
    #dist_params = distribution[4]
    #dist = getattr(stats, dist_params['name'])(*dist_params['params'])
    #rvs = dist.rvs(2400)
    #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    #axs[0].hist(rvs)
    #axs[1].hist(client_dataset[:, 4])
    #plt.show(block=True)

# Server generates data and trains model
generated_samples = np.zeros((10000, 23))
generated_sample_labels = np.zeros(10000)

for i in range(10):
    distributions = distribution_lists[i]
    samples = np.zeros((1000, 23))
    for dist_index, dist_params in enumerate(distributions):
        dist = getattr(stats, dist_params['name'])(*dist_params['params'])
        samples[:,dist_index] = dist.rvs(1000)
    
    sample_labels = np.zeros(1000)
    sample_labels = models[i].predict(samples)
    generated_samples[i*1000:(i+1)*1000] = samples
    generated_sample_labels[i*1000:(i+1)*1000] = np.argmax(sample_labels, axis=1)
    print(i+1)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(23,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'),
])
model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
#train_dataset = tf.data.Dataset.from_tensor_slices((generated_samples, generated_sample_labels))
#train_dataset = train_dataset.batch(32, drop_remainder=True)
model.fit(generated_samples, generated_sample_labels, batch_size=32, epochs=epochs)
_, accuracy = model.evaluate(test_set[:, :-1], test_set[:, -1], verbose=0)
print("Final accuracy="+ str(accuracy))
