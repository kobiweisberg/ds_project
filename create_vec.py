from dataloader import *
import opts
from models import DocEncoder
import torch
import time
import numpy as np
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def create_vec():

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=50,
                    help='number of epochs')
    parser.add_argument('--input_encoding_size', type=int, default=256, # 512
                        help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--rnn_size', type=int, default=256,  # 512
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--dropout', type=int, default=1,
                        help='1 for implementing dropout in lstm')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0,
    #                 help='weight_decay')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
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
    parser.add_argument('--filter_num', type=float, default=100,
                        help='number of cnn filters (from the same size')
    parser.add_argument('--model_name', type=str, default='',
                    help='model name')
    parser.add_argument('--load_model_name', type=str, default='/model_conv_5e-4.pth',
                        help='model name')
    parser.add_argument('--save_file', type=int, default=0,
                        help='save or don\'t save all encoded doc vectors')

    opt = parser.parse_args()



    load_model_name = opt.load_model_name  # model_conv or model
    print('model name is {}' .format(load_model_name))
    loader = Dataloader(opt)
    opt.vocab_size = len(loader.decoder)  # get vocab size
    encoded_docs, docs_length, labels = loader.only_encoded_docs[:], loader.docs_length[:], loader.labels[:]
    # load model
    if opt.cnn_model:
        model = DocEncoder.DocVec(opt)
    else:
        model = DocEncoder.DocEncoder(opt)
    model.cuda()
    # Assure in training mode
    model.eval()
    # load model
    checkpoint_path = opt.checkpoint_path + load_model_name
    model.load_state_dict(torch.load(checkpoint_path))
    print("model load from {}".format(checkpoint_path))
    vecs_rep_all = torch.zeros(len(docs_length),opt.filter_num).detach()
    start = time.time()
    for iteration in range(len(docs_length)):
            batch_docs, batch_masks, batch_labels, finished = get_batch(encoded_docs, docs_length, labels, opt, iteration)
            torch.cuda.synchronize()
            vec_rep = model(batch_docs, batch_masks, batch_labels, iteration, eval_flag=True)
            vecs_rep_all[iteration] = vec_rep
            torch.cuda.synchronize()
            if iteration % round(len(docs_length)/10)==0:
                print('finished {}/{} that\'s {} % ' .format(iteration+1, len(docs_length), 100*round((iteration+1)/len(docs_length),2)))
            # if iteration % round(len(docs_length) / 100) == 0:
            #     print('iteration is {} and time is {}' .format(iteration, time.time()-start))
    if opt.save_file:  # default is not saving
        np.save('vecs_rep_all.npy', vecs_rep_all.detach().numpy())
    return vecs_rep_all.detach().numpy()

#opt = opts.parse_opt()
#create_vec(opt)
create_vec()
