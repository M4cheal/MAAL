import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from data.fundusda_da import FundusDALoader
import logging
logger = logging.getLogger(__name__)
from utils.tools import *
from ever.util.param_util import count_model_parameters
from module.viz import VisualizeSegmm
from utils.dice_score import multiclass_dice_coeff
import torch.nn as nn
from torchvision import transforms
import cv2
from evaluate_mu import eval_print_log

def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None):
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir, palette)
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict,  strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)
    model.eval()
    print(cfg.EVAL_DATA_CONFIG)
    eval_dataloader = FundusDALoader(cfg.EVAL_DATA_CONFIG, is_eval=True)
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    dice_score = 0
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls, _ = model(ret)

            onehot_cls = F.one_hot(cls.argmax(dim=1), 3).permute(0, 3, 1, 2).float()
            onehot_cls_gt = F.one_hot(ret_gt['cls'].long(), 3).permute(0, 3, 1, 2).float().cuda()
            dice_score += multiclass_dice_coeff(onehot_cls[:, 1:, ...], onehot_cls_gt[:, 1:, ...],
                                               reduce_batch_first=False)
            probs = F.softmax(cls, dim=1)[0]
            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            full_mask = tf(probs.cpu()).squeeze()
            pre_mask = F.one_hot(full_mask.argmax(dim=0), 3).permute(2, 0, 1).numpy()
            out6 = torch.max(torch.tensor(pre_mask), 0)
            index6 = out6.indices
            n26 = index6.numpy()
            target = n26
            target = target.astype(np.uint8)
            new_label = np.zeros(target.shape, dtype=np.int64)
            new_label[target == 0] = 255
            new_label[target == 1] = 128
            new_label[target == 2] = 0
            target = new_label
            for fname, pred in zip(ret_gt['fname'], cls):
                cv2.imwrite(os.path.join(vis_dir, fname[:-4] + '.bmp'), target)

    logger.info('----[eval Dice] = {}'.format(dice_score/len(eval_dataloader)))
    metric_op.summary_all()
    torch.cuda.empty_cache()

    try:
        eval_print_log(vis_dir, str(cfg.EVAL_DATA_CONFIG.get('mask_dir')[0]), logger)
    except Exception as e:
        logger.info('!!!!!!!!![error] = {}'.format(e))

def evaluate_test(model, cfg, is_training=False, ckpt_path=None, logger=None):
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-test-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir, palette)
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict,  strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)
    model.eval()
    print(cfg.EVAL_DATA_CONFIG)
    eval_dataloader = FundusDALoader(cfg.EVAL_DATA_CONFIG, is_eval=True)
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    dice_score = 0
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls, _ = model(ret)

            onehot_cls = F.one_hot(cls.argmax(dim=1), 3).permute(0, 3, 1, 2).float()
            onehot_cls_gt = F.one_hot(ret_gt['cls'].long(), 3).permute(0, 3, 1, 2).float().cuda()
            dice_score += multiclass_dice_coeff(onehot_cls[:, 1:, ...], onehot_cls_gt[:, 1:, ...],
                                               reduce_batch_first=False)
            probs = F.softmax(cls, dim=1)[0]
            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            full_mask = tf(probs.cpu()).squeeze()
            pre_mask = F.one_hot(full_mask.argmax(dim=0), 3).permute(2, 0, 1).numpy()
            out6 = torch.max(torch.tensor(pre_mask), 0)
            index6 = out6.indices
            n26 = index6.numpy()
            target = n26
            target = target.astype(np.uint8)
            new_label = np.zeros(target.shape, dtype=np.int64)
            new_label[target == 0] = 255 # 背景
            new_label[target == 1] = 128 # OD
            new_label[target == 2] = 0 #OC
            target = new_label
            for fname, pred in zip(ret_gt['fname'], cls):
                cv2.imwrite(os.path.join(vis_dir, fname[:-4] + '.bmp'), target)


    logger.info('----[eval Dice] = {}'.format(dice_score/len(eval_dataloader)))
    metric_op.summary_all()
    torch.cuda.empty_cache()

    try:
        eval_print_log(vis_dir, str(cfg.EVAL_DATA_CONFIG.get('mask_dir')[0]), logger)
    except Exception as e:
        logger.info('!!!!!!!!![error] = {}'.format(e))


if __name__ == '__main__':
    seed_torch(2333)
    ckpt_path = r'...\REFUGE1000.pth'
    from nets.segformer_maal import SegFormer
    cfg = import_config('MAAL.adaptseg.REFUGEVAL.MAAL')
    model = SegFormer(num_classes=cfg.NUM_CLASSES, phi='b5', pretrained=False).cuda()
    logger = get_console_file_logger(name='SegFormer', logdir=cfg.SNAPSHOT_DIR)

    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict, strict=True)
    evaluate_test(model, cfg, False, ckpt_path, logger)