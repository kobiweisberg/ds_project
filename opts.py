import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=50,
                    help='number of epochs')
    parser.add_argument('--input_encoding_size', type=int, default=256, #128, #256, # 512
                        help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--rnn_size', type=int, default=256, #128, #256,  # 512
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--dropout', type=int, default=1,
                        help='1 for implementing dropout in lstm')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0,
    #                 help='weight_decay')
    parser.add_argument('--weight_decay', type=float, default=0,
                                         help='weight_decay')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--batch_size', type=int, default=1,  # for model without conv layer 30
                        help='minibatch size')
    parser.add_argument('--num_classes', type=int, default=20,
                        help='number of doc classes')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--save_checkpoint_every', type=int, default=1000,
                    help='how often to save a train loss history (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='/home/lab/vgilad/PycharmProjects/lstm_ds_project/checkpoint_history',
                    help='directory to store checkpointed models')
    parser.add_argument('--cnn_model', type=int, default=1,
                        help='1 for lstm with conv layer, 0 without conv layer')
    parser.add_argument('--filter_len', type=float, default=5,
                        help='length of cnn filter')
    parser.add_argument('--filter_num', type=float, default=200, #50 #100, # 200
                        help='number of cnn filters (from the same size')
    parser.add_argument('--model_name', type=str, default='model_conv_pp1_max_0.05_min_0.0001.pth',
                    help='model name')
    parser.add_argument('--load_model_name', type=str, default='/model_conv_pp1_max_0.05_min_0.0001.pth',
                        help='model name')
    parser.add_argument('--save_file', type=int, default=0,
                        help='save or don\'t save all encoded doc vectors')
    parser.add_argument('--cuda_flag', type=int, default=1,
                        help='if 1 use cuda')

    args = parser.parse_args()
    return args
