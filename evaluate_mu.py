import cv2
import numpy as np
import os
from evaluation_metrics_for_segmentation import *

def return_list(data_path,data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list
def eval_print_log(pred_img_path, gt_img_path, logger=None):

    file_list = return_list(pred_img_path, '.bmp')
    n = len(file_list)
    DC_score_cup = {'img_name': 'DC_cup score'}
    DC_score_disc = {'img_name': 'DC_disc score'}
    JAC_score_cup = {'img_name': 'JAC_cup score'}
    JAC_score_disc = {'img_name': 'JAC_disc score'}
    ACC_score_cup = {'img_name': 'ACC_cup score'}
    ACC_score_disc = {'img_name': 'ACC_disc score'}
    SEN_score_cup = {'img_name': 'SEN_cup score'}
    SEN_score_disc = {'img_name': 'SEC_disc score'}
    SPC_score_cup = {'img_name': 'SPC_cup score'}
    SPC_score_disc = {'img_name': 'SPC_disc score'}
    CDR_score = {'img_name': 'CDR score'}

    cup_dices = []
    disc_dices = []
    cup_JAC = []
    disc_JAC = []
    cup_ACC = []
    disc_ACC = []
    cup_SEN = []
    disc_SEN = []
    cup_SPC = []
    disc_SPC = []
    CDR = []

    DC_total = 0
    JC_total = 0
    ACC_total = 0
    SEN_total = 0
    SPC_total = 0
    CDR_total = 0

    for i in range(n):
        i = i
        temp_list = file_list[i]
        pred_name = os.path.join(pred_img_path, temp_list[:-4] + '.bmp')
        gt_name = os.path.join(gt_img_path, temp_list[:-4] + '.bmp')
        pred = cv2.imread(pred_name, 0)
        gt = cv2.imread(gt_name, 0)

        gt_oc = cv2.imread(gt_name, 0)
        gt_oc[gt_oc == 128] = 255

        gt_od = cv2.imread(gt_name, 0)
        gt_od[gt_od == 128] = 0

        dst_oc = cv2.Canny(gt_oc, 150, 200)
        dst_od = cv2.Canny(gt_od, 150, 200)

        pred_RGB = cv2.imread(pred_name)
        pred_RGB[dst_oc == 255] = [0, 0, 255]
        pred_RGB[dst_od == 255] = [0, 255, 0]

        cv2.imwrite(os.path.join(pred_img_path, 'TRUE_' + temp_list[:-4] + '.png'), pred_RGB)

        cup_dice, cup_jac, cup_acc, cup_sen, cup_spc, disc_dice, disc_jac, disc_acc, disc_sen, disc_spc, cdr = evaluate_binary_segmentation_1(
            pred, gt)
        DC_score_cup[temp_list] = cup_dice
        DC_score_disc[temp_list] = disc_dice
        JAC_score_cup[temp_list] = cup_jac
        JAC_score_disc[temp_list] = disc_jac
        ACC_score_cup[temp_list] = cup_acc
        ACC_score_disc[temp_list] = disc_acc
        SEN_score_cup[temp_list] = cup_sen
        SEN_score_disc[temp_list] = disc_sen
        SPC_score_cup[temp_list] = cup_spc
        SPC_score_disc[temp_list] = disc_spc
        CDR_score[temp_list] = cdr

        cup_dices.append(cup_dice)
        disc_dices.append(disc_dice)
        cup_JAC.append(cup_jac)
        disc_JAC.append(disc_jac)
        cup_ACC.append(cup_acc)
        disc_ACC.append(disc_acc)
        cup_SEN.append(cup_sen)
        disc_SEN.append(disc_sen)
        cup_SPC.append(cup_spc)
        disc_SPC.append(disc_spc)
        CDR.append(cdr)

    mean_cup_dice = np.mean(cup_dices)
    mean_disc_dice = np.mean(disc_dices)
    DC_score_cup[' DC_cup mean_score'] = mean_cup_dice
    DC_score_disc['DC_disc mean_score'] = mean_disc_dice
    mean_cup_jac = np.mean(cup_JAC)
    mean_disc_jac = np.mean(disc_JAC)
    JAC_score_cup[' JAC_cup mean_score'] = mean_cup_jac
    JAC_score_disc['JAC_disc mean_score'] = mean_disc_jac
    mean_cup_acc = np.mean(cup_ACC)
    mean_disc_acc = np.mean(disc_ACC)
    ACC_score_cup[' ACC_cup mean_score'] = mean_cup_acc
    ACC_score_disc['ACC_disc mean_score'] = mean_disc_acc
    mean_cup_sen = np.mean(cup_SEN)
    mean_disc_sen = np.mean(disc_SEN)
    SEN_score_cup[' SEM_cup mean_score'] = mean_cup_sen
    SEN_score_disc['SEN_disc mean_score'] = mean_disc_sen
    mean_cup_spc = np.mean(cup_SPC)
    mean_disc_spc = np.mean(disc_SPC)
    SPC_score_cup[' SPC_cup mean_score'] = mean_cup_spc
    SPC_score_disc['SPC_disc mean_score'] = mean_disc_spc

    mean_cdr = np.mean(CDR)
    CDR_score[' CDR mean_score'] = mean_cdr

    logger.info('DISC_DICE mean :' + str(mean_disc_dice))
    logger.info('CUP_DICE mean :' + str(mean_cup_dice))
    logger.info('DISC_JAC mean :' + str(mean_disc_jac))
    logger.info('CUP_JAC mean :' + str(mean_cup_jac))
    logger.info('DISC_ACC mean :' + str(mean_disc_acc))
    logger.info('CUP_ACC mean :' + str(mean_cup_acc))
    logger.info('DISC_SEN mean :' + str(mean_disc_sen))
    logger.info('CUP_SEN mean :' + str(mean_cup_sen))
    logger.info('DISC_SPC mean :' + str(mean_disc_spc))
    logger.info('CUP_SPC mean :' + str(mean_cup_spc))
    logger.info('CDR mean :' + str(mean_cdr))
