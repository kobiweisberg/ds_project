from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import cPickle
#from dataloader import *
import matplotlib.pyplot as plt
import argparse
import matplotlib.legend as lgd
import os
# get input directory
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str,
                    default='/home/lab/vgilad/PycharmProjects/lstm_ds_project/checkpoint_history',
                    help='get results from the "pkl" files in this input directory')
parser.add_argument('--output_dir', type=str,
                    default='/home/lab/vgilad/PycharmProjects/lstm_ds_project/plots/',
                    help='output directory to save the received plot')
"""
parser.add_argument('--input_dir', type=str,
                    default='./checkpoint_history',
                    help='get results from the "pkl" files in this input directory')
parser.add_argument('--output_dir', type=str,
                    default='../plots/',
                    help='output directory to save the received plot')
"""
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
history_file_name = 'histories.pkl'

# histories file



os.chdir(input_dir)

# change histories file name according to the input_dir
base_dir = os.path.basename(os.path.normpath(input_dir))

with open(history_file_name, 'rb') as history_f:
    history = cPickle.load(history_f)


train_accuracy, val_accuracy, val_loss, train_losses, train_iterations  = [ [] for i in range(5)]

for train_iter in history.keys():
    train_iterations.append(train_iter)
    train_losses.append(history[train_iter]['train_loss'])
    train_accuracy.append(history[train_iter]['train_accuracy'])	
    val_loss.append(history[train_iter]['val_loss'])
    val_accuracy.append(history[train_iter]['val_accuracy'])
    #val_losses.append(val_loss.cpu().numpy())

#  train loss
train_graph, = plt.plot(train_iterations, train_losses, '-o', label='train')
val_graph, = plt.plot(train_iterations, val_loss, '-o', label='val')
legend = plt.legend(fontsize=20, handles=[train_graph, val_graph], loc=1)

plt.title('Train Loss vs Iteration', fontsize=25)
plt.xlabel('iteration', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.savefig(output_dir + 'loss_vs_iteration.png')
plt.show()
plt.close()

#  accuracy
train_accuracy_graph, = plt.plot(train_iterations, train_accuracy, '-o', label='train')
val_accuracy_graph, = plt.plot(train_iterations, val_accuracy, '-o', label='val')
legend = plt.legend(fontsize=20, handles=[train_accuracy_graph, val_accuracy_graph], loc=4)

plt.title('Accuracy vs Iteration', fontsize=25)
plt.xlabel('iteration', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.savefig(output_dir + 'Accuracy_vs_iteration.png')
plt.show()


