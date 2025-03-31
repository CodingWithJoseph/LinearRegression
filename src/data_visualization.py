import matplotlib.pyplot as plt
import seaborn


def vis_correlation_matrix(bunch):
    matrix = bunch.frame.corr()
    plt.figure(figsize=(10, 8))
    seaborn.heatmap(data=matrix, annot=True)
    plt.show()


def vis_prediction_label(X, y, y_predictions):
    plt.scatter(X, y)
    plt.plot(X, y_predictions, color='red')
    plt.xlabel('MedInc')
    plt.ylabel('MedHouseVal')
    plt.show()
