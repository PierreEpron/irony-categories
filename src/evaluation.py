from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef

import numpy as np

ID_2_LABEL = {
        0:"No Irony",
        1:"Irony by Clash",
        2:"Situational Irony",
        3:"Other Irony"
}

##### Fleiss Kappa ######


def convert_to_A(y):
    ''' Convert given list of label for Task A fleiss kappa (0=0, {1,2,3}=1) '''
    return [0 if v == 0 else 1 for v in y]

def convert_to_B(y):
    ''' Convert given list of label for Task B fleiss kappa (0=0, 1=1, {2,3}=2) '''
    return [v if v < 2 else 2 for v in y]

def fleiss_kappa(y_true, y_pred):
    ''' 
        Compute fleiss kappa score for a given list of true labels and predicted labels.
        This is not an complete fleiss kappa implementation. It only work for two "annotators".
    '''
    N = len(y_true)
    k = len(set(y_true))

    x = np.zeros((N, k))
    for _k in range(k):
        x[:, _k] = (np.array(y_true) == _k).astype(int) + (np.array(y_pred) == _k).astype(int)

    n = x[0].sum()
    pi = np.sum(1 / (n*(n-1)) * (np.sum(x**2, axis=1) - n)) / x.shape[0]
    pj = np.sum((np.sum(x, axis=0) / (n * x.shape[0])) ** 2)
    kappa = (pi-pj) / (1-pj) 

    return kappa

def get_A_B_fleiss_kappa(y_true, y_pred):
    ''' 
        Compute and return fleiss kappa for task A and task B
    '''
    return (
        fleiss_kappa(convert_to_A(y_true), convert_to_A(y_pred)),
        fleiss_kappa(convert_to_B(y_true), convert_to_B(y_pred))
    )

##### Evaluation Report #####

def eval_report(y_true, y_pred, id_2_label=ID_2_LABEL):
    ''' Shortcut to report all metrics for a given list of true labels and predicted labels'''

    print(classification_report(y_true, y_pred, target_names=list(id_2_label.values())))
    print("MCC:", matthews_corrcoef(y_true, y_pred))
    fleiss_A, fleiss_B = get_A_B_fleiss_kappa(y_true, y_pred)
    print("Fleiss K:")
    print("\tTask A:", fleiss_A)
    print("\tTask B:", fleiss_B)
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=list(id_2_label.values())).plot(xticks_rotation="vertical") 

