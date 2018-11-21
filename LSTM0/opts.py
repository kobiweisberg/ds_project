import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=30,
                    help='number of epochs')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                        help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--dropout', type=int, default=1,
                        help='1 for implementing dropout in lstm')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='minibatch size')
    parser.add_argument('--num_classes', type=int, default=20,
                        help='number of doc classes')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--save_checkpoint_every', type=int, default=100,
                    help='how often to save a train loss history (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='/home/lab/vgilad/PycharmProjects/lstm_ds_project/checkpoint_history',
                    help='directory to store checkpointed models')

    args = parser.parse_args()
    return args
