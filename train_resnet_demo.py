import argparse
import datetime
import os
import os.path as osp
import pytz

import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import train
from config import configurations



def get_log_dir(model_name, config_id, cfg, verbose=True):
    # Creates an output directory for each experiment, timestamped
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    if verbose:
	    for k, v in cfg.items():
	        v = str(v)
	        if '/' in v:
	            continue
	        name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


here = osp.dirname(osp.abspath(__file__)) # output folder is located here


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', default='resnet_face_demo')
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('-d', '--dataset_path', 
                        default='./samples/tiny_dataset')
    parser.add_argument('-m', '--model_path', default=None)
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir(args.exp_name, args.config, cfg, verbose=False)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size    


    # -----------------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------------
    #  Images should be arranged like this:
    #   data_root/
    #       class_1/....jpg..
    #       class_2/....jpg.. 
    #       ......./....jpg.. 
    data_root = args.dataset_path
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    
    # Data transforms
    # http://pytorch.org/docs/master/torchvision/transforms.html
    transform = transforms.Compose([
        transforms.Scale(256),  # smallest size resizes to 256
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ]),
    ])

    # Data loader
    # http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
    traindir = args.dataset_path
    train_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(traindir, transform), 
                    batch_size=3, shuffle=True, **kwargs)

    # for demo purpose, set to be same as train
    val_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(traindir, transform), 
                    batch_size=2, shuffle=False, **kwargs) 

    print 'dataset classes:' + str(train_loader.dataset.classes)
    num_class = len(train_loader.dataset.classes)


    # -----------------------------------------------------------------------------
    # 2. Model
    # -----------------------------------------------------------------------------
    # PyTorch ResNet model definition: 
    #   https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    # ResNet docs:
    #   http://pytorch.org/docs/master/torchvision/models.html#id3
    model = torchvision.models.resnet50(pretrained=True) # Using pre-trained for demo purpose

    # by default, resnet has 1000 output categories
    print model.fc  # Check: Linear (2048 -> 1000)
    model.fc = torch.nn.Linear(2048, num_class) # change to current dataset's classes
    print model.fc
    
    if args.model_path:
        # If a PyTorch model is to be loaded from a file
        checkpoint = torch.load(args.model_path)        
        model.load_state_dict(checkpoint['model_state_dict'])

    start_epoch = 0
    start_iteration = 0

    if resume:
        # Resume training from last saved checkpoint
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        pass

    # Loss - cross entropy between predicted scores (unnormalized) and class labels
    # http://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#crossentropyloss
    criterion = nn.CrossEntropyLoss()

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()


    # -----------------------------------------------------------------------------
    # 3. Optimizer
    # -----------------------------------------------------------------------------
    params = filter(lambda p: p.requires_grad, model.parameters()) 
    # parameters with p.requires_grad=False are not updated during training
    # this can be specified when defining the nn.Modules during model creation

    if 'optim' in cfg.keys():
    	if cfg['optim'].lower()=='sgd':
    		optim = torch.optim.SGD(params,
				        lr=cfg['lr'],
				        momentum=cfg['momentum'],
				        weight_decay=cfg['weight_decay'])
    	elif cfg['optim'].lower()=='adam':
    		optim = torch.optim.Adam(params,
				        lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    	else:
    		raise NotImplementedError('Optimizers: SGD or Adam')
    else:
	    optim = torch.optim.SGD(params,
			        lr=cfg['lr'],
			        momentum=cfg['momentum'],
			        weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])


    # -----------------------------------------------------------------------------
    # [optional] Sanity-check: forward pass with a single batch
    # -----------------------------------------------------------------------------
    DEBUG = False
    if DEBUG:   
        dataiter = iter(val_loader)
        img, label = dataiter.next()

        print 'Labels: ' + str(label.size()) # batchSize x num_class
        print 'Input: ' + str(img.size())    # batchSize x 3 x 224 x224

        im = img.squeeze().numpy()
        im = im[0,:,:,:]    # get first image in the batch
        im = im.transpose((1,2,0)) # permute to 224x224x3
        f = plt.figure()
        plt.imshow(im)
        plt.savefig('sanity-check-im.jpg')  # save transformed image in current folder

        inputs = Variable(img)
        if cuda:
            inputs = inputs.cuda()

        model.eval()
        outputs = model(inputs)
        print 'Network output: ' + str(outputs.size())

        model.train()
    else:
        pass


    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    trainer = train.Trainer(
        cuda=cuda,
        model=model,
        criterion=criterion,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()

