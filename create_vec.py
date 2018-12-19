from dataloader import *
import opts
from models import DocEncoder
import torch
import time
import numpy as np
import argparse
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def create_vec(opt, make_plot=False):


    n_start = None #1000#None#1000#200
    n_stop = None #-8500#None#-8500 #  -2000
    load_model_name = opt.load_model_name  # model_conv or model
    print('model name is {}' .format(load_model_name))
    loader = Dataloader(opt)
    super_class_labels_names = loader.super_class_labels_by_name
    opt.vocab_size = len(loader.decoder)  # get vocab size
    # original
    encoded_docs, docs_length, labels = loader.only_encoded_docs[n_start:n_stop], loader.docs_length[n_start:n_stop], loader.labels[n_start:n_stop]
    #encoded_docs, docs_length, labels = loader.only_encoded_docs, loader.docs_length, loader.labels

    # load model
    if opt.cnn_model:
        model = DocEncoder.DocVec(opt)
    else:
        model = DocEncoder.DocEncoder(opt)
    if opt.cuda_flag:
        model.cuda()
    else:
        model.cpu()

    # Assure in evaluation mode
    model.eval()
    # load model
    checkpoint_path = opt.checkpoint_path + load_model_name
    vecs_rep_all = torch.ones(len(docs_length), opt.filter_num).detach()  # original
    if opt.save_file:
        model.load_state_dict(torch.load(checkpoint_path))
        print("model load from {}".format(checkpoint_path))
        #vecs_rep_all = torch.ones(len(docs_length),opt.filter_num).detach()  # original
        #vecs_rep_all = torch.ones(len(docs_length), 20).detach()  # embed size 20
        #vecs_rep_all = []
        #vecs_rep_all = np.zeros([len(docs_length), opt.filter_num])
        start = time.time()
        for iteration in range(len(docs_length)):
            if opt.save_file:
                #batch_docs, batch_masks, batch_labels, finished = get_batch(encoded_docs, docs_length, labels, opt, iteration)
                torch.cuda.synchronize()
                #vec_rep = model(batch_docs, batch_masks, batch_labels, iteration, eval_flag=True)  # original
                label = torch.from_numpy(np.asarray(labels[iteration]))
                encoded_doc = torch.from_numpy(np.asarray(encoded_docs[iteration])).unsqueeze(0)
                vec_rep = model(encoded_doc, None, label, opt, iteration, eval_flag=True)
                #vec_rep = model((torch.tensor(encoded_docs[iteration])).unsqueeze(0), 0, labels[iteration], iteration, eval_flag=True)
                # from sklearn.preprocessing import normalize
                # norm1 = vec_rep.cpu().numpy() / np.linalg.norm(vec_rep.cpu().numpy())
                # vecs_rep_all[iteration] = norm1
                vecs_rep_all[iteration,:] = vec_rep  # original
                #vecs_rep_all.append(vec_rep)
                torch.cuda.synchronize()
                if iteration % round(len(docs_length)/10)==0:
                    print('finished {}/{} that\'s {} % ' .format(iteration+1, len(docs_length), 100*round((iteration+1)/len(docs_length),2)))
                # if iteration % round(len(docs_length) / 100) == 0:
                #     print('iteration is {} and time is {}' .format(iteration, time.time()-start))
            else:
                break
    labels_names = [loader.target_names[x] for x in labels]
    super_class_labels_names = loader.super_class_labels_by_name
    vecs_rep_all = vecs_rep_all.detach().numpy()
    embedding_size_str = '_model_conv_pp_kobi_17_12_18_max_0.05_min_0.0001'
    if opt.save_file:  # default is not saving
        np.save('/home/lab/vgilad/PycharmProjects/lstm_ds_project/files/vecs_rep_all' + embedding_size_str + '.npy', vecs_rep_all)
        np.save('/home/lab/vgilad/PycharmProjects/lstm_ds_project/files/labels_names' + embedding_size_str, np.asarray(labels_names))
    else:
        vecs_rep_all = np.load('/home/lab/vgilad/PycharmProjects/lstm_ds_project/files/vecs_rep_all' + embedding_size_str + '.npy')
        labels_names = np.load('/home/lab/vgilad/PycharmProjects/lstm_ds_project/files/labels_names.npy')
        print('vecs and labels are loaded')
    if make_plot:
        #corpus = Dataloader(args)
        #stop = len(labels_names) + n_start
        #plot_tsne(vecs_rep_all[n_start:stop-1], (labels_names[1:len(labels_names)], super_class_labels_names[n_start+1:stop]), seed=4,perplexity=30, alpha=0.3)
        plot_tsne(vecs_rep_all[:], (labels_names[:],super_class_labels_names[:]), seed=4, perplexity=30, alpha=0.3)  # original
        #plot_tsne(vecs_rep_all[:2000], (labels_names[:2000], super_class_labels_names[:2000]), seed=4,perplexity=30, alpha=0.3)

        #plot_tsne(np.stack(vecs_rep_all.squeeze(1), 0), labels_names, seed=4, perplexity=30, alpha=0.3)
        #plot_tsne(vecs_rep_all, labels_names, seed=4, perplexity=30, alpha=0.3)
    #return vecs_rep_all.detach().numpy()  # original
    return vecs_rep_all[1000:-2000], labels_names[1000:-2000]

#opt = opts.parse_opt()
#create_vec(opt)
#create_vec()

if __name__=='__main__':
    #from LM_vectorizer import plot_tsne

    opt = opts.parse_opt()
    create_vec(opt, make_plot=False)
    #vecs = create_vec(opt, make_plot=True)

