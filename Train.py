import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import os.path as osp
from module.Discriminator import Discriminator_MWPD
from data.fundusda_da import FundusDALoader
from ever.core.iterator import Iterator
from utils.tools import *
from tqdm import tqdm
from eval import evaluate
from torch import autograd
import torch.nn as nn
from nets.segformer_maal import SegFormer
from utils.dice_score import dice_loss
parser = argparse.ArgumentParser(description='Run MAAL methods.')

# parser.add_argument('--config_path', default='MAAL.adaptseg.DS.MAAL', type=str, help='config path')
parser.add_argument('--config_path', default='MAAL.adaptseg.REFUGEVAL.MAAL', type=str, help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)


def main():
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='MAAL', logdir=cfg.SNAPSHOT_DIR)
    # Create network
    model = SegFormer(num_classes=cfg.NUM_CLASSES, phi='b5', pretrained=True)

    model.train()
    model.cuda()
    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)

    # init D
    model_D1 = Discriminator_MWPD(cfg.NUM_CLASSES * 5)

    model_D1.train()
    model_D1.cuda()

    count_model_parameters(model, logger)
    count_model_parameters(model_D1, logger)

    trainloader = FundusDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    targetloader = FundusDALoader(cfg.TARGET_DATA_CONFIG)
    targetloader_iter = Iterator(targetloader)

    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)


    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, betas=(0.9, 0.99))
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()


    source_label = 0
    target_label = 1
    one = torch.FloatTensor(1)
    mone = one * -1
    one = one.cuda().mean()
    mone = mone.cuda().mean()
    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0
        Wasserstein_D = 0

        optimizer.zero_grad()
        G_lr = adjust_learning_rate(optimizer, i_iter, cfg)

        optimizer_D1.zero_grad()
        D_lr = adjust_learning_rate_D(optimizer_D1, i_iter, cfg)

        for sub_i in range(cfg.ITER_SIZE):
            # train G
            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            # train with source

            batch = trainloader_iter.next()
            images, labels = batch[0]
            images = Variable(images).cuda()

            pred1, s4_pred1 = model(images)

            loss_seg1 = loss_calc(pred1, labels['cls'].cuda())
            loss_dice = dice_loss(F.softmax(pred1, dim=1).float(),
                                  F.one_hot(
                                      torch.from_numpy(labels['cls'].clone().numpy().astype(np.int64)).cuda(),
                                      3).permute(0, 3, 1, 2).float(),
                                  multiclass=True)
            loss = loss_seg1 + 0.1 * loss_dice

            # proper normalization
            loss = loss / cfg.ITER_SIZE
            loss.backward()
            loss_seg_value1 += loss.data.cpu().numpy()

            # train with target
            batch = targetloader_iter.next()
            images, labels = batch[0]
            images = Variable(images).cuda()

            pred_target1, s4_pred_target1 = model(images)

            D_out1 = model_D1(torch.cat([pred_target1, s4_pred_target1], dim=1))
            D_out1 = cfg.LAMBDA_ADV_TARGET2 * D_out1.mean()
            D_out1.backward(mone)

            loss_adv_target_value1 -= D_out1.data.cpu().numpy() / cfg.ITER_SIZE

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True


            # train with source
            pred1 = pred1.detach()
            s4_pred1 = s4_pred1.detach()

            D_out1 = model_D1(torch.cat([pred1, s4_pred1], dim=1))
            D_out1 = D_out1.mean()
            D_out1.backward(mone)

            loss_D_value1 += D_out1.data.cpu().numpy()

            # train with target
            pred_target1 = pred_target1.detach()
            s4_pred_target1 = s4_pred_target1.detach()

            D_out1 = model_D1(torch.cat([pred_target1, s4_pred_target1], dim=1))
            D_out1 = D_out1.mean()
            D_out1.backward(one)

            for p in model_D1.parameters():
                p.data.clamp_(-0.01, 0.01)

            loss_D_value1 -= D_out1.data.cpu().numpy()

            Wasserstein_D -= loss_D_value1
        optimizer.step()
        optimizer_D1.step()

        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info(
                'iter = %d loss_seg1 = %.3f  G_score = %.3f,  D_score = %.3f, Wasserstein_D = %.3f,  G_lr = %.5f D_lr = %.5f' % (
                    i_iter, loss_seg_value1, loss_adv_target_value1, loss_D_value1, Wasserstein_D, G_lr, D_lr)
            )

        if i_iter >= cfg.NUM_STEPS_STOP - 1 or i_iter >= cfg.NUM_STEPS_EARLY_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter+1) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model_D1.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter+1) + '_D1.pth'))
            evaluate(model, cfg, True, ckpt_path, logger)
            break

        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model_D1.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '_D1.pth'))
            evaluate(model, cfg, True, ckpt_path, logger)
            model.train()


if __name__ == '__main__':
    seed_torch(2333)
    main()
