import argparse

import torch
from torch.autograd import Variable

#import data
from LSTM0.dataloader import *
#from LM_hagai import repackage_hidden

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

parser = argparse.ArgumentParser(description='Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./LSTM0/files/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pth',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f,map_location='cpu').to(device)
model.eval()

corpus = Dataloader(args)
ntokens = len(corpus.decoder)
def batchify(corpus, bsz):
    c = []
    for data in corpus:
        data = torch.LongTensor(data)
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        c.append(data.view(bsz, -1).t().contiguous().to(device))
    return c
data = batchify(corpus.only_encoded_docs, 1)

def get_docs_repr(model,data):
    docs_representation = []
    with torch.no_grad():
        for i,doc in enumerate(data):
            hidden = model.init_hidden(1)
            hidden = repackage_hidden(hidden)
            out, before_lin, hidden = model(doc, hidden)
            assert before_lin.size(1) == 1
            doc_repr = before_lin.mean(dim=1).mean(dim=0)
            docs_representation.append(doc_repr.cpu().numpy())

    return docs_representation





