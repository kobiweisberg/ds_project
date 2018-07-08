import numpy as np
import pickle
import matplotlib.pyplot as plt
def evaluate_many2one(mat,num_of_labels):
    mat = mat[:num_of_labels,:]
    correct_pred = np.zeros(num_of_labels)
    for col in mat.T:
        correct_pred[np.argmax(col)] +=np.max(col)

    return correct_pred/np.sum(mat,axis=1)


if __name__=='__main__':
    with open('results/','rb') as rf:
        mat = pickle.load(rf)
    ret = evaluate_many2one(mat,20)
    plt.figure(1)
    plt.plot(range(1,21),ret,'*')
    plt.show()