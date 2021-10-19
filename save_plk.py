from __future__ import print_function
import os
import collections
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from data.datamgr import SimpleDataManager
from learn2capture import configs
import wrn_model
from learn2capture.io_utils import parse_args, get_resume_file

use_gpu = torch.cuda.is_available()

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 
    def forward(self, x):
        return self.module(x)

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def extract_feature(val_loader, model, checkpoint_dir, tag='last',set='novel_cd'):
    save_dir = '{}/{}'.format(checkpoint_dir, tag)
    if os.path.isfile(save_dir + '/%s_features.plk'%set):
        data = load_pickle(save_dir + '/%s_features.plk'%set)
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    #model.eval()
    with torch.no_grad():
        
        output_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(val_loader):
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs,_ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)
    
        all_info = output_dict
        save_pickle(save_dir + '/%s_features_mini2CUB.plk'%set, all_info)
        return all_info

if __name__ == '__main__':
    params = parse_args('test')
    params.model = 'WideResNet28_10'
    params.method = 'S2M2_R'

    loadfile_base = configs.data_dir[params.dataset] + 'base.json'
    loadfile_novel = configs.data_dir[params.dataset] + 'novel.json'   # configs.data_dir[params.dataset]
    if params.dataset == 'miniImagenet' or params.dataset == 'CUB':
        datamgr       = SimpleDataManager(84, batch_size = 256)
    base_loader = datamgr.get_data_loader(loadfile_base, aug=False)
    novel_loader      = datamgr.get_data_loader(loadfile_novel, aug = False)

    checkpoint_dir = '%s/checkpoints/%s' %(configs.save_dir, 'miniImagenet')  # params.dataset  mini->CUB
    # checkpoint_dir = r'D:\CZQ\ICCV2021-DCM-Metrics\checkpoints\miniImagenet\255.tar'  # params.dataset  mini->CUB
    modelfile   = get_resume_file(checkpoint_dir)

    if params.model == 'WideResNet28_10':
        model = wrn_model.wrn28_10(num_classes=params.num_classes)


    model = model.cuda()
    cudnn.benchmark = True

    checkpoint = torch.load(modelfile)
    state = checkpoint['state']
    state_keys = list(state.keys())

    callwrap = False
    if 'module' in state_keys[0]:
        callwrap = True
    if callwrap:
        model = WrappedModel(model)
    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    model.eval()
    # output_dict_base = extract_feature(base_loader, model, checkpoint_dir, tag='last', set='base')
    # print("base set features saved!")
    output_dict_novel=extract_feature(novel_loader, model, checkpoint_dir, tag='last',set='novel')
    print("novel features saved!")
