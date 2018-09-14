from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, log_nth=0 , num_epochs=10):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)


        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        num_iterations = num_epochs * iter_per_epoch
        k = 0
        l = 0
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            correctval=0
            totalval=0
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                k += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model.forward(inputs)
                numBatches, numClasses = outputs.size()
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()

                # print statistics

                self.train_loss_history.append(loss.data[0])

                _, predicted = torch.max(outputs.data, 1)
                total += inputs.size(0)
                correct += (predicted == labels.data).sum()
                train_acc = 100*correct/total

                self.train_acc_history.append(train_acc)

                if k==50:
                    print('[%d] loss_train: %.3f, train_acc: %.3f' %(epoch + 1, loss.data[0], train_acc))
                    k = 0

            for t, data in enumerate(val_loader):
                l += 1
                # get the inputs
                inputsval, labelsval = data

                # wrap them in Variable
                inputsval, labelsval = Variable(inputsval), Variable(labelsval)

                # forward + backward + optimize
                outputsval = model.forward(inputsval)
                numBatchesval, numClassesval = outputsval.size()
                criterionval = torch.nn.CrossEntropyLoss()
                lossval = criterionval(outputsval, labelsval)

                # print statistics

                self.val_loss_history.append(lossval.data[0])

                _, predicted = torch.max(outputsval.data, 1)
                totalval += inputs.size(0)
                correctval += (predicted == labels.data).sum()
                val_acc = 100*correctval/totalval

                self.val_acc_history.append(val_acc)

                if l==50:
                    print('[%d] loss_val: %.3f, val_acc: %.3f' %(epoch + 1, lossval.data[0], val_acc))
                    l = 0

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
