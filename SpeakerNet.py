#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from accuracy import accuracy
from DatasetLoader import loadWAV
from loss.protoloss import ProtoLoss


class SpeakerNet(nn.Module):

    def __init__(self, max_frames, lr = 0.0001, model="alexnet50", nOut = 512, optimizer = 'adam', encoder_type = 'CAP', global_clf = True, **kwargs):
        super(SpeakerNet, self).__init__();

        self.global_clf = global_clf
        self.encoder_type = encoder_type

        self.alpha = 1
        self.beta = 1
        self.fc_dim = nOut

        argsdict = {'nOut': nOut, 'encoder_type':encoder_type}

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__(model)
        self.__S__ = SpeakerNetModel(**argsdict).cuda();

        self.__L__ = ProtoLoss().cuda()

        if optimizer == 'adam':
            self.__optimizer__ = torch.optim.Adam(self.parameters(), lr = lr);
            self.__scheduler__ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer__, patience=3, threshold =1e-4)
        elif optimizer == 'sgd':
            self.__optimizer__ = torch.optim.SGD(self.parameters(), lr = lr, momentum = 0.9, weight_decay=1e-4, nesterov=True);
            self.__scheduler__ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer__, patience=6, threshold =1e-4)
        else:
            raise ValueError('Undefined optimizer.')
        

        self.__max_frames__ = max_frames;

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy

        criterion = torch.nn.CrossEntropyLoss()
        
        for data, data_label in loader:

            tstart = time.time()

            self.zero_grad();

            feat = []
            for idx,inp in enumerate(data):
                if idx == 0: # support set
                    outp      = self.__S__.forward(inp.cuda())
                    feat_s = outp
                else:            # query set
                    outp      = self.__S__.forward(inp.cuda())
                    feat.append(outp)
            feat_q = torch.cat(feat)
            n_q = len(data)-1
            label   = torch.LongTensor(data_label).cuda()

            if self.encoder_type == 'TAP':
                spk_s = self.__S__.TAP(feat_s)
                spk_q = self.__S__.TAP(feat_q)

            if self.encoder_type == 'SAP':
                spk_s = self.__S__.SAP(feat_s)
                spk_q = self.__S__.SAP(feat_q)

            if self.encoder_type =='CAP':
                spk_s, spk_q= self.__S__.CAP(feat_s, feat_q)
                logit_e = torch.sum(spk_q * F.normalize(spk_s, dim=2), dim=2)
                label_e = torch.from_numpy(numpy.asarray(range(0,stepsize))).repeat(n_q).cuda()   
                loss_e = F.cross_entropy(logit_e, label_e)
                prec1, _ = accuracy(logit_e.detach().cpu(), label_e.detach().cpu(), topk=(1, 5))
                prec1 = prec1.item()
                if self.global_clf: #Global classification
                    eye = torch.eye(stepsize).repeat(n_q,1).unsqueeze(2).cuda()
                    spk_s = (spk_s * eye).sum(dim=1)
                    spk_q = (spk_q * eye).sum(dim=1)
                    cat_input = torch.cat((spk_s, spk_q), dim=0)

                    logit_g = F.linear(cat_input, F.normalize(self.__S__.global_w, dim=1))
                    labels = label.repeat(2*n_q)
                    loss_g = F.cross_entropy(logit_g, labels)
            else:
                loss_e, prec1 = self.__L__(spk_s, spk_q)
                prec1 = prec1.item()
                if self.global_clf: #Global classification
                    cat_input = torch.cat((spk_s,spk_q), dim=0)
                    logit_g = F.linear(cat_input, F.normalize(self.__S__.global_w, dim=1))
                    labels = label.repeat(1+n_q)
                    loss_g = F.cross_entropy(logit_g, labels)
                else:
                    loss_g = 0
            nloss = self.alpha * loss_e + self.beta * loss_g 
            self.__optimizer__.zero_grad()
            nloss.backward()
            self.__optimizer__.step()
            loss    += nloss.detach().cpu();
            top1    += prec1;
            counter += 1;
            index   += stepsize;
            telapsed = time.time() - tstart
            sys.stdout.write("\rProcessing (%d/%d) "%(index, loader.nFiles));
            sys.stdout.write("Loss %f EER/T1 %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
            sys.stdout.write("Q:(%d/%d)"%(loader.qsize(), loader.maxQueueSize));
            sys.stdout.flush();

        sys.stdout.write("\n");
        
        return (loss/counter, top1/counter);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Read data from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def readDataFromList(self, listfilename):

        data_list = {};

        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if not line:
                    break;

                data = line.split();
                filename = data[1];
                speaker_name = data[0]

                if not (speaker_name in data_list):
                    data_list[speaker_name] = [];
                data_list[speaker_name].append(filename);

        return data_list


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromListSave(self, listfilename, print_interval=5000, feat_dir='', test_path='', num_eval=10, eval_frames=None):
        self.eval();
        
        lines       = []
        files       = []
        filedict    = {}
        feats       = {}
        tstart      = time.time()

        if feat_dir != '':
            print('Saving temporary files to %s'%feat_dir)
            if not(os.path.exists(feat_dir)):
                os.makedirs(feat_dir)

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line): #  or (len(all_scores)==1000) 
                    break;

                data = line.split();

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval).cuda()

            ref_feat = self.__S__.forward(inp1)
            if self.encoder_type == 'TAP':
                ref_feat = self.__S__.TAP(ref_feat)
            elif self.encoder_type == 'SAP':
                ref_feat = self.__S__.SAP(ref_feat)

            ref_feat = ref_feat.detach().cpu()
            filename = '%06d.wav'%idx

            if feat_dir == '':
                feats[file]     = ref_feat
            else:
                filedict[file]  = filename
                torch.save(ref_feat,os.path.join(feat_dir,filename))

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d: %.2f Hz, embed size %d"%(idx,idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            if feat_dir == '':
                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()
            else:
                ref_feat = torch.load(os.path.join(feat_dir,filedict[data[1]])).cuda()
                com_feat = torch.load(os.path.join(feat_dir,filedict[data[2]])).cuda()

            if self.encoder_type =='CAP':
                ref_feat, com_feat= self.__S__.CAP(ref_feat, com_feat)
                score = torch.sum(F.normalize(ref_feat, dim=2)* F.normalize(com_feat, dim=2), dim=2).mean().detach().cpu().numpy()
            else:
                score = torch.matmul(F.normalize(ref_feat, dim=-1), F.normalize(com_feat, dim=-1).T).mean().detach().cpu().numpy()

            all_scores.append(score);  
            all_labels.append(int(data[0]));

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d: %.2f Hz"%(idx,idx/telapsed));
                sys.stdout.flush();

        if feat_dir != '':
            print(' Deleting temporary files.')
            shutil.rmtree(feat_dir)

        print('\n')

        return (all_scores, all_labels);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Update learning rate
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def updateLearningRate(self, alpha):

        learning_rate = []
        for param_group in self.__optimizer__.param_groups:
            param_group['lr'] = param_group['lr']*alpha
            learning_rate.append(param_group['lr'])

        return learning_rate;


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

