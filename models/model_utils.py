import torch
import torch.nn.functional as F

import numpy as np

def train_helper(model, optimizer, scheduler,
                 epoch, data_loader, use_cuda, args=None):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.type(torch.FloatTensor)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if args is not None and args.basic_block == 'fc':
          data = data.reshape(len(data), -1)
        output = model(data)
        if output.shape[1] == 1:
          # regression
          output = output.view(-1)
          loss = F.smooth_l1_loss(output, target)
        else:
          if use_cuda:
            target = target.type(torch.cuda.LongTensor)
          else:
            target = target.type(torch.LongTensor)
          # classification
          output = F.log_softmax(output, dim=1)
          loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.data))

def test_helper(model, data_loader, use_cuda, is_test, args=None):
    model.eval()
    loss = 0.0
    correct = 0.0

    ys, preds = [], []
    num_samples = 0

    for data, target in data_loader:
        data = data.type(torch.FloatTensor)
        num_samples += len(data)

        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            if args is not None and args.basic_block == 'fc':
              data = data.reshape(len(data), -1)
            output = model(data)
            if output.shape[1] == 1:
              # regression
              output = output.view(-1)
              loss += F.smooth_l1_loss(output, target, reduction='sum').data
              preds += output.cpu().numpy(),
              ys += target.cpu().numpy(),
            else:
              # classification
              if use_cuda:
                target = target.type(torch.cuda.LongTensor)
              else:
                target = target.type(torch.LongTensor)

              output = F.log_softmax(output, dim=1)
              loss += F.nll_loss(output, target, reduction='sum').data

              pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

              correct += pred.eq(target.view_as(pred)).sum().data

    loss /= num_samples

    if len(output.shape) == 1:
      print('\nData set stats: Average loss: {:.4f}\n'.format(loss))
      return loss, 0

    else:
      if use_cuda:
          correct = correct.cpu()
      acc = 100.0 * correct.numpy() / num_samples
      print('\nData set stats: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
          loss, correct, len(data_loader.dataset), acc))

      return loss, 100.0 - acc


def greedy_train_helper(model, optimizer, scheduler,
                 epoch, data_loader, use_cuda):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.type(torch.FloatTensor), target.type(torch.LongTensor)
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        # prepare the target variables
        past_data =  model.past_feature_map(data)
        features = past_data[0].data
        features.requires_grad = True
        preds = past_data[1](features)

        output = F.log_softmax(preds, dim = 1)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        reg_target = features.grad.data

        # start the main optimization
        optimizer.zero_grad()
        output = model(data)
        loss = torch.sum(reg_target*output) + + 1e-5*torch.norm(output)**2
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())


        if batch_idx % 100 == 0:
            print('(Regression) Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.data))


def test_helper_adaBoost(model, iteration, data_loader, use_cuda, is_test):
    model.eval()
    correct = 0.0

    batch_size = 0
    num_samples = 0
    for data, target in data_loader:
        data, target = data.type(torch.FloatTensor), target.type(torch.LongTensor)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            pred = model.get_predictions(data, iteration + 1)
            correct += pred.eq(target.view_as(pred)).sum().data
        if batch_size == 0:
            batch_size = len(data)
        num_samples += len(data)

    if use_cuda:
        correct = correct.cpu()
    acc = 100.0 * correct.numpy() / num_samples
    print('\nData set stats: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(data_loader.dataset), acc))

    return 100.0 - acc

