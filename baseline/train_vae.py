import argparse
import time
import os
import random
import collections
import numpy as np
import torch
import pickle

from model import DAE, VAE, AAE
from vocab import Vocab
from meter import AverageMeter
from utils import set_seed, logging, load_sent


import sys

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--train', metavar='FILE', required=True,
                    help='path to training file')
parser.add_argument('--valid', metavar='FILE', default='',
                    help='path to validation file (optional)')
parser.add_argument('--save-dir', default='checkpoints', metavar='DIR',
                    help='directory to save checkpoints and outputs')
parser.add_argument('--load-model', default='', metavar='FILE',
                    help='path to load checkpoint if specified')
# Architecture arguments
parser.add_argument('--vocab-size', type=int, default=10000, metavar='N',
                    help='keep N most frequent words in vocabulary')
parser.add_argument('--dim_z', type=int, default=128, metavar='D',
                    help='dimension of latent variable z')
# changed embedding dim from 512 to 768
parser.add_argument('--dim_emb', type=int, default=768, metavar='D',
                    help='dimension of word embedding')
parser.add_argument('--dim_h', type=int, default=1024, metavar='D',
                    help='dimension of hidden state per layer')
parser.add_argument('--nlayers', type=int, default=1, metavar='N',
                    help='number of layers')
parser.add_argument('--dim_d', type=int, default=512, metavar='D',
                    help='dimension of hidden state in AAE discriminator')
# Model arguments
parser.add_argument('--model_type', default='dae', metavar='M',
                    choices=['dae', 'vae', 'aae'],
                    help='which model to learn')
parser.add_argument('--lambda_kl', type=float, default=0, metavar='R',
                    help='weight for kl term in VAE')
parser.add_argument('--lambda_adv', type=float, default=0, metavar='R',
                    help='weight for adversarial loss in AAE')
parser.add_argument('--lambda_p', type=float, default=0, metavar='R',
                    help='weight for L1 penalty on posterior log-variance')
parser.add_argument('--noise', default='0,0,0,0', metavar='P,P,P,K',
                    help='word drop prob, blank prob, substitute prob'
                         'max word shuffle distance')
# Training arguments
parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP',
                    help='dropout probability (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
#parser.add_argument('--clip', type=float, default=0.25, metavar='NORM',
#                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
# Others
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
# added parser arguments
parser.add_argument('--append-results', action='store_true',
                    help='append results to existing files instead of overwriting')

if np.__version__ < '2.0.0':
    # For NumPy 1.x, map numpy.core to numpy._core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._exceptions'] = getattr(np.core, '_exceptions', None)


def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs in batches:
            batch_size = inputs.size(0)
            inputs = inputs.unsqueeze(0)  
            losses = model.autoenc(inputs, inputs)  
            for k, v in losses.items():
                meters[k].update(v.item(), batch_size)
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters


def load_embedding_pkl(path):
    # return embeddings
    embeddings =[]
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def get_embedding_batches(embeddings, batch_size, device):
    batches = []
    # compute number of padding elements
    remainder = len(embeddings) % batch_size
    if remainder != 0:
        padding = torch.zeros(batch_size - remainder, *embeddings.shape[1:])
        embeddings = torch.cat([embeddings, padding])
    
    batches = torch.split(embeddings, batch_size)
    batches = [torch.as_tensor(x) for x in batches]
    batches = torch.stack(batches)
    return batches

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)

    # Prepare data
    train_embs = load_embedding_pkl(args.train)
    train_embs = [torch.as_tensor(x) for x in train_embs]
    train_embs = torch.stack(train_embs) 

    original_sample_count = train_embs.size(0)  # record original sample count
    logging('# train samples {}'.format(original_sample_count), log_file)
    
    # # only load if validation set provided
    if args.valid:
        valid_embs = load_embedding_pkl(args.valid)
        valid_batches = get_embedding_batches(valid_embs, args.batch_size, device)
        logging('# valid samples {}'.format(len(valid_embs)), log_file)
    else:
        valid_batches = None
    
    vocab_file = os.path.join(args.save_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        Vocab.build(train_embs, vocab_file, args.vocab_size)
    vocab = Vocab(vocab_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)
    device = torch.device('cpu')
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[args.model_type](
        vocab, args).to(device)
    if args.load_model:
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model'])
        model.flatten()
    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)

    train_batches = get_embedding_batches(train_embs, args.batch_size, device)
  
    # add one dimension
    train_batches = train_batches.unsqueeze(1)

    best_val_loss = None
    all_train_losses = []
    all_vector_losses = []  

    for epoch in range(args.epochs):
        start_time = time.time()
        logging('-' * 80, log_file)
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(train_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            inputs = train_batches[idx]  

            batch_size = inputs.size(0)
            
            # compute actual sample count (excluding padding)
            start_sample_idx = idx * args.batch_size
            end_sample_idx = min(start_sample_idx + batch_size, original_sample_count)
            actual_batch_size = end_sample_idx - start_sample_idx
            
            # skip if this batch is all padding
            if actual_batch_size <= 0:
                continue

            inputs_for_model = inputs

            # forward and backward pass
            losses = model.autoenc(inputs_for_model, inputs_for_model, is_train=True)
            loss = model.loss(losses)
            model.step(losses)
            meters['loss'].update(loss.item())
            
            # record average loss per batch
            batch_losses = [loss.item()]
            all_train_losses.extend(batch_losses)
            
            # compute vector loss (vector, not scalar)
            with torch.no_grad():
                model.eval()
                # get reconstructed embeddings
                mu, logvar, z, reconstructed = model(inputs_for_model, is_train=False)
                
                input_embeddings = inputs_for_model  
                output_embeddings = reconstructed   

                # compute vector_loss for actual samples (excluding padding)
                actual_inputs = input_embeddings[:, :actual_batch_size, :]  
                actual_outputs = output_embeddings[:, :actual_batch_size, :] 

                # compute vector diff
                vector_diffs = actual_inputs - actual_outputs

                batch_avg_vector_diff = torch.mean(vector_diffs, dim=1)  
                batch_avg_vector_diff = batch_avg_vector_diff.squeeze(0)  
                
                all_vector_losses.append(batch_avg_vector_diff.cpu().numpy())

           

            if (i + 1) % args.log_interval == 0:
                # compute L2 norm of vector diff for display
                vector_norm = np.linalg.norm(all_vector_losses[-1]) if all_vector_losses else 0.0
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches | loss {:.2f} | vector_norm {:.4f}'.format(
                    epoch + 1, i + 1, len(indices), meters['loss'].avg, vector_norm)
                meters['loss'].clear()
                logging(log_output, log_file)

        
        if valid_batches is not None:
            valid_meters = evaluate(model, valid_batches)
            logging('-' * 80, log_file)
            log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
                epoch + 1, time.time() - start_time)
            for k, meter in valid_meters.items():
                log_output += ' {} {:.2f},'.format(k, meter.avg)
            if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
                log_output += ' | saving model'
                ckpt = {'args': args, 'model': model.state_dict()}
                torch.save(ckpt, os.path.join(args.save_dir, 'model.pt'))
                best_val_loss = valid_meters['loss'].avg
            logging(log_output, log_file)
        else:
            ckpt = {'args': args, 'model': model.state_dict()}
            torch.save(ckpt, os.path.join(args.save_dir, 'model.pt'))
            logging('| end of epoch {:3d} | time {:5.0f}s | saved model'.format(
                epoch + 1, time.time() - start_time), log_file)
    
    logging('Done training', log_file)
    
    all_vector_losses = np.array(all_vector_losses)
    vector_file_path = os.path.join(args.save_dir, 'vector_losses.npy')
    
    if args.append_results and os.path.exists(vector_file_path):
        # append mode
        existing_losses = np.load(vector_file_path)
        combined_losses = np.concatenate([existing_losses, all_vector_losses], axis=0)
        np.save(vector_file_path, combined_losses)
        logging(f'Appended to existing file. New shape: {combined_losses.shape}', log_file)
    else:
        # overwrite mode (default)
        np.save(vector_file_path, all_vector_losses)
        logging(f'Saved new file. Shape: {all_vector_losses.shape}', log_file)

    logging('Saved training losses, vector losses and texts to {}'.format(args.save_dir), log_file)


if __name__ == '__main__':
    args = parser.parse_args()
    args.noise = [float(x) for x in args.noise.split(',')]
    main(args)
    