import numpy as np
import matplotlib.pyplot as plt

AROUSAL_MIN, AROUSAL_MAX = 1, 9
VALENCE_MIN, VALENCE_MAX = 1, 9
DEG = 5


def predict(arousal_means, valence_means):
    l = len(arousal_means)
    arousal_predict = np.poly1d(np.polyfit(range(1, l+1), arousal_means, 3))
    valence_predict = np.poly1d(np.polyfit(range(1, l+1), valence_means, 3))

    arousal_val = arousal_predict(l+1)
    valence_val = valence_predict(l+1)

    print(f'arousal predict: {arousal_val}, valence predict: {valence_val}')

    arousal_means = np.append(arousal_means, arousal_val)
    valence_means = np.append(valence_means, valence_val)
    return arousal_means, valence_means


def main(data, num_times=10):
    data = data.T
    arousal_mean, valence_mean = data[0], data[1]
    for _ in range(num_times):
        arousal_mean, valence_mean = predict(arousal_mean, valence_mean)
    for i in range(10):
        arousal_mean[~i] = max(AROUSAL_MIN, min(AROUSAL_MAX, arousal_mean[~i]))
        valence_mean[~i] = max(VALENCE_MIN, min(VALENCE_MAX, valence_mean[~i]))

    x, y = arousal_mean, valence_mean
    t = list(range(1, len(x)+1))

    plt.plot(x, y)
    plt.scatter(x, y, color=["g"]*data.shape[1] + ["b"]*num_times)
    for i, txt in enumerate(t):
        plt.annotate(f' {txt}', (x[i], y[i]))
    plt.xlabel('arousal mean')
    plt.ylabel('valence mean')
    plt.title('Prediction of next 10 songs based on the first five songs')
    plt.savefig('prediction_next_10', dpi=300)


data = np.array([
    [0.8, 4.1],
    [1.4, 3.6],
    [2.2, 3.1],
    [2.9, 2.7],
    [3.3, 2],
    [4, 1.4]
])
main(data)



