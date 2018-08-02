import numpy as np


def evaluate_many2one(mat, num_of_labels):
    mat = mat[:num_of_labels,:]
    correct_pred = np.zeros(num_of_labels)
    for col in mat.T:
        correct_pred[np.argmax(col)] +=np.max(col)

    return correct_pred/np.sum(mat,axis=1)


def evaluate_many2one_conf_mat(mat, num_of_labels):
    mat = mat[:num_of_labels, :]
    labes_conf_mat = np.zeros((num_of_labels,num_of_labels))
    for col in mat.T:
        predicted_label = np.argmax(col)
        labes_conf_mat[:,predicted_label] += col

    return labes_conf_mat


def conf_mat2scores(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = []
    for i in range(cm.shape[0]):
        temp = np.delete(cm, i, 0)  # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    accuracy = np.sum(TP)/np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy,precision,recall

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import pickle
    '''with open('results/','rb') as rf:
        mat = pickle.load(rf)
    ret = evaluate_many2one(mat,20)
    plt.figure(1)
    plt.plot(range(1,21),ret,'*')
    plt.show()'''
    a = np.array([[38,2,200],[47,53,4]])
    print(evaluate_many2one_conf_mat(a,2))