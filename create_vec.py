from dataloader import *
import opts
from models import DocEncoder
import torch
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def create_vec(opt):

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

opt = opts.parse_opt()
create_vec(opt)
