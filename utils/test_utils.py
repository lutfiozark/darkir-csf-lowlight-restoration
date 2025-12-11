import sys, os
import numpy as np
import torch
import torch.distributed as dist
from lpips import LPIPS
from tqdm import tqdm
from torchvision.transforms import ToPILImage

sys.path.append('../losses')
sys.path.append('../data/datasets/datapipeline')
from losses import *

calc_SSIM = SSIM(data_range=1.)
to_pil = ToPILImage()

#---------- Set of functions to work with DDP
def setup(rank, world_size, Master_port='12355', backend=None):
    """
    Initialize the process group with a backend that works on the current setup.
    Defaults to NCCL when available, otherwise falls back to Gloo (needed for CPU/Windows).
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = Master_port

    if backend is None:
        use_nccl = torch.cuda.is_available() and dist.is_nccl_available()
        backend = 'nccl' if use_nccl else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def _extract_state_dict(ckpt):
    """
    Support loading checkpoints saved in different formats.
    """
    if isinstance(ckpt, dict):
        for key in ('params', 'model_state_dict', 'model', 'state_dict'):
            if key in ckpt:
                return ckpt[key]
    return ckpt

def load_checkpoint(model, path_weights, map_location='cpu', strict=True):
    """
    Loads checkpoints that may contain 'params', 'model_state_dict' or a raw state_dict.
    Handles the DDP 'module.' prefix automatically.
    """
    ckpt = torch.load(path_weights, map_location=map_location, weights_only=False)
    state_dict = _extract_state_dict(ckpt)

    has_prefix = all(k.startswith('module.') for k in state_dict.keys())
    target = model if has_prefix else getattr(model, 'module', model)
    target.load_state_dict(state_dict, strict=strict)
    return model

def save_model(model, path):
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), path)

def shuffle_sampler(samplers, epoch):
    '''
    A function that shuffles all the Distributed samplers in the loaders.
    '''
    if not samplers: # if they are none
        return
    for sampler in samplers:
        sampler.set_epoch(epoch)

def _save_samples(low, enhanced, gt, save_dir, idx, max_samples):
    """
    Save a few sample triplets to disk for visualization.
    """
    if save_dir is None or idx >= max_samples:
        return
    os.makedirs(save_dir, exist_ok=True)
    base = f"{idx:05d}"
    to_pil(torch.clamp(low, 0., 1.)).save(os.path.join(save_dir, f"{base}_input.png"))
    to_pil(torch.clamp(enhanced, 0., 1.)).save(os.path.join(save_dir, f"{base}_output.png"))
    if gt is not None:
        to_pil(torch.clamp(gt, 0., 1.)).save(os.path.join(save_dir, f"{base}_gt.png"))

def eval_one_loader(model, test_loader, metrics, rank=0, world_size = 1, eta = False, save_dir=None, max_save=8):
    calc_LPIPS = LPIPS(net = 'vgg', verbose=False).to(rank)
    mean_metrics = {'valid_psnr':[], 'valid_ssim':[], 'valid_lpips':[]}

    if eta: pbar = tqdm(total = int(len(test_loader)))
    with torch.no_grad():
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for idx, (high_batch_valid, low_batch_valid) in enumerate(test_loader):

            high_batch_valid = high_batch_valid.to(rank)
            low_batch_valid = low_batch_valid.to(rank)         

            enhanced_batch_valid = model(low_batch_valid)
            # loss
            valid_loss_batch = torch.mean((high_batch_valid - enhanced_batch_valid)**2)
            valid_ssim_batch = calc_SSIM(enhanced_batch_valid, high_batch_valid)
            valid_lpips_batch = calc_LPIPS(enhanced_batch_valid, high_batch_valid)
                
            valid_psnr_batch = 20 * torch.log10(1. / torch.sqrt(valid_loss_batch))        
    
            mean_metrics['valid_psnr'].append(valid_psnr_batch.item())
            mean_metrics['valid_ssim'].append(valid_ssim_batch.item())
            mean_metrics['valid_lpips'].append(torch.mean(valid_lpips_batch).item())

            if dist.get_rank() == 0:
                _save_samples(low_batch_valid[0].cpu(), enhanced_batch_valid[0].cpu(), high_batch_valid[0].cpu(),
                              save_dir, idx, max_save)

            if eta: pbar.update(1)

    valid_psnr_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_psnr'])).to(rank), world_size=world_size)
    valid_ssim_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_ssim'])).to(rank),world_size=world_size)
    valid_lpips_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_lpips'])).to(rank), world_size=world_size)

    metrics['valid_psnr'] = valid_psnr_tensor.item()
    metrics['valid_ssim'] = valid_ssim_tensor.item()
    metrics['valid_lpips'] = valid_lpips_tensor.item()
    
    
    imgs_dict = {'input':low_batch_valid[0], 'output':enhanced_batch_valid[0], 'gt':high_batch_valid[0]}
    
    if eta: pbar.close()
    return metrics, imgs_dict

def eval_model(model, test_loader, metrics, rank=None, world_size = 1, eta = False, save_dir=None, max_save=8):
    '''
    This function runs over the multiple test loaders and returns the whole metrics.
    '''
    #first you need to assert that test_loader is a dictionary
    if type(test_loader) != dict:
        test_loader = {'data': test_loader}
    if len(test_loader) > 1:
        all_metrics = {}
        all_imgs_dict = {}
        for key, loader in test_loader.items():

            all_metrics[f'{key}'] = {}
            metrics, imgs_dict = eval_one_loader(model, loader['loader'], all_metrics[f'{key}'], rank=rank, world_size=world_size, eta=eta, save_dir=save_dir, max_save=max_save)
            all_metrics[f'{key}'] = metrics
            all_imgs_dict[f'{key}'] = imgs_dict
        return all_metrics, all_imgs_dict
    
    else:
        metrics, imgs_dict = eval_one_loader(model, test_loader['data'], metrics, rank=rank, world_size=world_size, eta=eta, save_dir=save_dir, max_save=max_save)
        return metrics, imgs_dict

def eval_one_loader_two_models(model1, model2, test_loader, metrics, devices = ['cuda:0', 'cuda:1'], eta = False):
    calc_LPIPS = LPIPS(net = 'vgg', verbose=False).to(devices[0])
    mean_metrics = {'valid_psnr':[], 'valid_ssim':[], 'valid_lpips':[]}

    if eta: pbar = tqdm(total = int(len(test_loader)))
    with torch.no_grad():
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for high_batch_valid, low_batch_valid in test_loader:

            high_batch_valid = high_batch_valid.to(devices[0])
            low_batch_valid = low_batch_valid.to(devices[0])         

            enhanced_batch_valid = model1(low_batch_valid)
            enhanced_batch_valid = torch.clamp(enhanced_batch_valid, 0., 1.)
            enhanced_batch_valid = model2(enhanced_batch_valid.to(devices[1]))
            # loss
            enhanced_batch_valid = enhanced_batch_valid.to(devices[0])
            valid_loss_batch = torch.mean((high_batch_valid - enhanced_batch_valid)**2)
            valid_ssim_batch = calc_SSIM(enhanced_batch_valid, high_batch_valid)
            valid_lpips_batch = calc_LPIPS(enhanced_batch_valid, high_batch_valid)
                
            valid_psnr_batch = 20 * torch.log10(1. / torch.sqrt(valid_loss_batch))        
            # print(valid_loss_batch)
            mean_metrics['valid_psnr'].append(valid_psnr_batch.item())
            mean_metrics['valid_ssim'].append(valid_ssim_batch.item())
            mean_metrics['valid_lpips'].append(torch.mean(valid_lpips_batch).item())
            # print(valid_psnr_batch.item())
            if eta: pbar.update(1)
    print(mean_metrics['valid_psnr'])
    valid_psnr_tensor = np.mean(mean_metrics['valid_psnr'])
    valid_ssim_tensor = np.mean(mean_metrics['valid_ssim'])
    valid_lpips_tensor = np.mean(mean_metrics['valid_lpips'])

    metrics['valid_psnr'] = valid_psnr_tensor.item()
    metrics['valid_ssim'] = valid_ssim_tensor.item()
    metrics['valid_lpips'] = valid_lpips_tensor.item()
    
    
    imgs_dict = {'input':low_batch_valid[0], 'output':enhanced_batch_valid[0], 'gt':high_batch_valid[0]}
    
    if eta: pbar.close()
    return metrics, imgs_dict

def eval_model_two_models(model1, model2, test_loader, metrics, devices=['cuda:0', 'cuda:1'], eta = False):
    '''
    This function runs over the multiple test loaders and returns the whole metrics.
    '''
    #first you need to assert that test_loader is a dictionary
    if type(test_loader) != dict:
        test_loader = {'data': test_loader}
    if len(test_loader) > 1:
        all_metrics = {}
        all_imgs_dict = {}
        for key, loader in test_loader.items():

            all_metrics[f'{key}'] = {}
            metrics, imgs_dict = eval_one_loader_two_models(model1, model2, loader['loader'], all_metrics[f'{key}'], devices = devices, eta=eta)
            all_metrics[f'{key}'] = metrics
            all_imgs_dict[f'{key}'] = imgs_dict
        return all_metrics, all_imgs_dict
    
    else:
        metrics, imgs_dict = eval_one_loader_two_models(model1, model2, test_loader['data'], metrics, devices = devices, eta=eta)
        return metrics, imgs_dict
