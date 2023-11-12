import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from diaglib.learn.utils.utils import *
import os
import time
import datetime
import torch.nn.functional as F
#from datasets.dataset_generic import save_splits
import diaglib.learn.models_vit.vision_transformer as vision_transformer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from timm.utils import accuracy, AverageMeter
from lr_scheduler import build_scheduler 
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import sys
#from utils.gpu_utils import gpu_profile, print_gpu_mem
#os.environ['GPU_DEBUG']='0'

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    if cur > 1: 
        print('\nTraining Fold {}!'.format(cur))
    else:
        print('\nTraining on a single fold!')
        
    if args.log_data:
        logger = args.logger
        
        if dist.get_rank() == 0:
            path = os.path.join(args.result_dir, "args.json")
            with open(path, "w") as f:
                f.write(args.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(args.dump())
        logger.info(args.dumps(vars(args)))
        
    if args.use_tensorboard: 
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None
        
    # create a directory for a current fold results/files/data 
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    print('\nInit train/val/test splits...', end=' ')
    if len(datasets) > 2:
        train_split, test_split, val_split = datasets
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))
    else: 
        train_split, val_split = datasets
        test_split = None
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))    
    
    print('\nInit Model...', end=' ')
    if args.n_classes is None: 
        args.n_classes = train_split.get_num_classes()
        print(f'----- assigning number of classes by calculating: {args.n_classes} -----')
    model_dict = {'image_size': args.patch_size, 'num_classes': args.n_classes}
    
    # if dropout specified, update model dict settings - otherwise just let it be default (so 0.0) 
    if args.drop_out is not None:
        model_dict.update({'dropout' : args.drop_out})
    
    # same for attention dropout 
    if args.att_dropout is not None: 
        model_dict.update({'attention_dropout':args.att_dropout})
    
    if args.model_type == 'vit_base_b_16':
        if args.pretrained:
            model = vision_transformer.base_vit_b_16(**model_dict, pretrained=True)
        else: 
            model = vision_transformer.base_vit_b_16(**model_dict)
    elif args.model_type == 'vit_base_b_32':
        if args.pretrained:
            model = vision_transformer.base_vit_b_32(**model_dict, pretrained=True)
        else: 
            model = vision_transformer.base_vit_b_32(**model_dict)
    elif args.model_type == 'vit_base_l_16':
        if args.pretrained:
            model = vision_transformer.base_vit_l_16(**model_dict, pretrained=True)
        else: 
            model = vision_transformer.base_vit_l_16(**model_dict)
    elif args.model_type == 'vit_base_l_32':
        if args.pretrained:
            model = vision_transformer.base_vit_l_32(**model_dict, pretrained=True)
        else: 
            model = vision_transformer.base_vit_l_32(**model_dict)
 
    # logging section 
    if args.log_data:
        logger.info(f"Creating model:{args.model_type} - pretrained: {args.pretrained}")
        logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model, 'flops'):
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")
        
    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.local_rank, broadcast_buffers=False)
        print('\nGPU available, using CUDA..')
    else: 
        print('\nGPU unavailable, using CPU..')
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, args, training=True, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split, args)
    if test_split is not None:
        test_loader = get_split_loader(test_split, args)
    print('Done!')

    if args.acc_steps > 1:
        lr_scheduler = build_scheduler(args, optimizer, len(train_loader) // args.acc_steps)
    else:
        lr_scheduler = build_scheduler(args, optimizer, len(train_loader))
    
    print('\nInit loss function...', end=' ')
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        loss_fn = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.:
        loss_fn = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing )
    else:
        loss_fn = nn.CrossEntropyLoss() # DEFAULT AND USED 
    
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 10, stop_epoch=30, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    max_accuracy = 0.0
    
    if args.log_data:
        logger.info("Start training")
        
    for epoch in range(args.max_epochs):

        train_loop(epoch, model, train_loader, optimizer, args.n_classes, lr_scheduler, logger, loss_fn, args.en_autocast, args.acc_steps, args.use_autocast, args.max_epochs, writer)
        stop, acc1, acc5, _ = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir, args.use_autocast, args.en_autocast, logger)
        
        logger.info(f"Accuracy of the network on the {epoch} epoch: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        
        if stop: 
            logger.info(f'Early stopping initialized on: {epoch} epoch.')
            print(f'Early stopping initialized on: {epoch} epoch.')
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    if test_split is not None:
        sum_result, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    else:
        sum_result = None
        test_auc = None
        val_auc = None

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        
        if test_split is not None:
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return sum_result, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop(epoch, model, loader, optimizer, n_classes, lr_scheduler, logger = None, loss_fn = None, en_autocast = True, acc_steps = 1, use_autocast = True, clip_grad=None, max_epochs = 300, writer = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # set scaler 
    scaler = torch.cuda.amp.GradScaler()
    
    # set model to train mode 
    model.train()
    optimizer.zero_grad()
    num_steps = len(loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    acc_logger = Accuracy_Logger(n_classes=n_classes) # do i even need this? 
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, batch in enumerate(loader):

        # get data and laber from given batch 
        data, label = batch
        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
        
        print(f'-- there are {len(model(data))} outputs from model --')
        
        # set norm to None for default (for logging only) ## default value
        norm = None
        
        # if autocast set to True - use it with scaler for handle FP16 problems  
        if use_autocast:
            with torch.cuda.amp.autocast(enabled=en_autocast):
                output = model(data) 
                loss = loss_fn(output, label)
                # divide loss by accumulation steps for grandient accumulation 
                loss = loss / acc_steps
            
            # get model y_hat 
            Y_hat = torch.topk(output, 1, dim = 1)[1]
            
            loss_value = loss.item() # just for logging
            acc_logger.log(Y_hat, label) # same as above 
            error = calculate_error(Y_hat, label) # same as above again
            scaler.scale(loss).backward() # loss backward with scaler 
            
            # step and update taking on account gradient accumulation
            if ((batch_idx + 1) % acc_steps == 0) or (batch_idx + 1 == len(loader)):
                
                if clip_grad is not None: 
                    # unscale before clipping
                    scaler.unscale_(optimizer)
                    norm  = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step_update((epoch * num_steps + batch_idx) // acc_steps)
                optimizer.zero_grad()       
        
        # if there is no autocast set - continue with default way 
        else: 
            # get outputs and loss for current loop 
            output = model(data)
            Y_hat = torch.topk(output, 1, dim = 1)[1]
            acc_logger.log(Y_hat, label) # just logging 
            error = calculate_error(Y_hat, label) # same as above 
            loss = loss_fn(output, label)
            
            # temporary "debbuging" :))
            print(o_one[3])
            o_one = output[0]
            print(loss)
            
            # for gradients accumulation - divide the gradients by the steps & backward
            loss = loss / acc_steps
            loss_value = loss.item()
            loss.backward()
            
            # for gradients accumulation - 
            if ((batch_idx + 1) % acc_steps == 0) or (batch_idx + 1 == len(loader)):
                optimizer.step()
                lr_scheduler.step_update((epoch * num_steps + batch_idx) // acc_steps)
                optimizer.zero_grad()       
        
        torch.cuda.synchronize() # just for time record (GPU is asynchronous)
        
        # gather data for logs and print current stats 
        if norm is not None:  # loss_scaler return None if not update
            norm_meter.update(norm)
        
        # same as above 
        train_loss += loss_value
        train_error += error
        loss_meter.update(loss.item(), label.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # for every 20 epochs print stats 
        if (batch_idx + 1) % 10 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - batch_idx)
            logger.info(
                f'Train: [{epoch}/{max_epochs}][{batch_idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
            
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    # print stats and add data to tensorboard
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer is not None:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, use_autocast = True, en_autocast = True, logger = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    end = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # get data and laber from given batch 
            data, label = batch
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            if use_autocast:
                with torch.cuda.amp.autocast(enabled=en_autocast):
                    output = model(data) 
            else: 
                output = model(data) 
            
            loss = loss_fn(output, label)
            
            # accuracy counter 
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), label.size(0))
            acc1_meter.update(acc1.item(), label.size(0))
            acc5_meter.update(acc5.item(), label.size(0))
            
            # for logging  and stats 
            Y_hat = torch.topk(output, 1, dim = 1)[1]
            Y_prob = F.softmax(output, dim = 1)
            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False, acc1_meter.avg, acc5_meter.avg, loss_meter.avg

def summary(model, loader, n_classes, use_autocast, en_autocast):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_path = loader.slide_data['patch_path']
    slide_subidx = loader.slide_data['subindex']
    slide_results = {}
    
    for batch_idx, batch in enumerate(loader):
        # get data and laber from given batch 
        data, label = batch
        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
        curr_path = slide_path[batch_idx]
        curr_subidx = slide_subidx[batch_idx]
        
        with torch.no_grad():
            if use_autocast:
                with torch.cuda.amp.autocast(enabled=en_autocast):
                    output = model(data) 
            else: 
                output = model(data) 

        Y_hat = torch.topk(output, 1, dim = 1)[1]
        Y_prob = F.softmax(output, dim = 1)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        error = calculate_error(Y_hat, label)
        test_error += error
        slide_results.update({curr_path: {'slide_path': f'{curr_path}[{curr_subidx}]', 'prob': probs, 'label': label.item()}})

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                print(calc_auc(fpr, tpr))
                aucs.append(calc_auc(fpr, tpr))
            else:
                print('nan')
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return slide_results, test_error, auc, acc_logger
