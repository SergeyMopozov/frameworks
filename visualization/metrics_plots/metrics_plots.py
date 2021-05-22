import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

def classification_scorer_roc_plot(model, X_valid, y_valid):
    """

    :param model:
    :param X_valid:
    :param y_valid:
    :return:
    """
    probabilities_valid = model.predict_proba(X_valid)
    probabilities_one_valid = probabilities_valid[:, 1]
    y_predict = model.predict(X_valid)

    auc_roc = roc_auc_score(y_valid, probabilities_one_valid)
    f1 = f1_score(y_valid, y_predict)

    print(f'AUC ROC: {auc_roc}')
    print(f'f1 score {f1}')



    fpr, tpr, thresholds = roc_curve(y_valid, probabilities_one_valid)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.show()


def classification_scorer(model, X_valid, y_valid):
    """

    :param model:
    :param X_valid:
    :param y_valid:
    :return:
    """
    plot_confusion_matrix(model, X_valid, y_valid)
    plt.show()
    print(classification_report(y_valid, model.predict(X_valid)))