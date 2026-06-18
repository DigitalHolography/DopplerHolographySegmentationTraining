import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utils
# -------------------------

def _ensure_shape(pred, targ):
    if targ.ndim == 3:
        targ = targ.unsqueeze(1)
    return pred.float(), targ.float()


def soft_skeletonize(img, thresh_width=10):
    for _ in range(thresh_width):
        min_pool = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
        img = torch.relu(img - torch.relu(img - min_pool))
    return img

# -------------------------
# Base losses
# -------------------------

class BCELoss:
    """Multi-label BCE loss (replacement for CE)"""
    def __init__(self, pos_weight=None):
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self, pred, targ):
        pred, targ = _ensure_shape(pred, targ)
        return self.loss_func(pred, targ)

    def activation(self, x): return torch.sigmoid(x)
    def decodes(self, x): return (torch.sigmoid(x) > 0.5).float()

class DiceLoss:
    """Multi-label Dice loss"""
    def __init__(self, smooth=1e-6):
        self.smooth = smooth

    def __call__(self, pred, targ):
        pred, targ = _ensure_shape(pred, targ)

        probs = torch.sigmoid(pred)

        intersection = (probs * targ).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targ.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class FocalLoss:
    """Multi-label focal loss"""
    def __init__(self, gamma=2.0, pos_weight=None):
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self, pred, targ):
        pred, targ = _ensure_shape(pred, targ)

        bce = self.bce_loss(pred, targ)
        pt = torch.exp(-bce)

        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()
    
# -------------------------
# Combined losses
# -------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, pos_weight=None, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] - raw outputs from model
        targets: [B, C, H, W] - binary masks (0 or 1)
        """
        # BCE Loss
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        logits = logits.float()
        targets = targets.float()
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='mean')
        bce = bce_loss_fn(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)
        num = (probs * targets).sum(dim=(2, 3))  # per batch & class
        den = (probs + targets).sum(dim=(2, 3))

        dice_score = (2 * num + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss
class FocalDiceLoss:
    def __init__(self, smooth=1., alpha=1., gamma=2.0):
        self.focal = FocalLoss(gamma)
        self.dice = DiceLoss(smooth)
        self.alpha = alpha

    def __call__(self, pred, targ):
        return self.focal(pred, targ) + self.alpha * self.dice(pred, targ)

    def activation(self, x): return torch.sigmoid(x)
    def decodes(self, x): return (torch.sigmoid(x) > 0.5).float()

class CLDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, cl_weight=0.5, dice_weight=0.3, bce_weight=0.2, pos_weight=None):
        super().__init__()
        self.smooth = smooth
        self.cl_weight = cl_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        logits = logits.float()
        targets = targets.float()

        probs = torch.sigmoid(logits)
        bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits, targets)

        # Standard Dice Loss
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()

        # clDice
        pred_skeleton = soft_skeletonize(probs)
        gt_skeleton = soft_skeletonize(targets)

        tprec = ((pred_skeleton * targets).sum(dim=(2, 3)) + self.smooth) / (pred_skeleton.sum(dim=(2, 3)) + self.smooth)
        tsens = ((gt_skeleton * probs).sum(dim=(2, 3)) + self.smooth) / (gt_skeleton.sum(dim=(2, 3)) + self.smooth)

        cl_dice_loss = 1 - (2 * tprec * tsens / (tprec + tsens + self.smooth)).mean()

        return self.cl_weight * cl_dice_loss + self.dice_weight * dice_loss + self.bce_weight * bce
    
    # -------------------------
    # TopoLoss
    # -------------------------

    import gudhi as gd
from pylab import *
import torch

def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """

    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None
        gt_n_holes = 0
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list()
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list()
    else:
        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1 it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect]
        else:
            idx_holes_perfect = list()

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes]

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:]

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)
    
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()

    lh_cubic = gd.CubicalComplex(
        dimensions=[lh.shape[0], lh.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return 0, 0, 0, False

    # return persistence diagram, birth/death critical points
    pd_lh = np.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
    bcp_lh = np.array([[pairs_lh[0][0][i][0]//lh.shape[1], pairs_lh[0][0][i][0]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    dcp_lh = np.array([[pairs_lh[0][0][i][1]//lh.shape[1], pairs_lh[0][0][i][1]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])

    return pd_lh, bcp_lh, dcp_lh, True

def getTopoLoss(likelihood_tensor, gt_tensor, topo_size=100):
    """
    Calculate the topology loss of the predicted image and ground truth image 
    Warning: To make sure the topology loss is able to back-propagation, likelihood 
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood_tensor:   The likelihood pytorch tensor.
        gt_tensor        :   The groundtruth of pytorch tensor.
        topo_size        :   The size of the patch is used. Default: 100

    Returns:
        loss_topo        :   The topology loss value (tensor)

    """

    likelihood = torch.sigmoid(likelihood_tensor).clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):
            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                         x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                         x:min(x + topo_size, gt.shape[1])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

            # print("pd_lh.shape", pd_lh.shape, "bcp_lh.shape", bcp_lh.shape, "dcp_lh.shape", dcp_lh.shape, "pairs_lh_pa", pairs_lh_pa)

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue
            if (pd_lh.shape[0] == 0 or pd_gt.shape[0] == 0): continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)

            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                            bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                for hole_indx in idx_holes_to_remove:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to death  # push to diagonal
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to birth # push to diagonal
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
    return loss_topo

class CETopoloss(nn.Module):
    def __init__(self, smooth=1e-6, coeff = 0.1, pos_weight=None):
        super().__init__()
        self.smooth = smooth
        self.lambda_ = coeff
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] - raw outputs from model
        targets: [B, C, H, W] - binary masks (0 or 1)
        """
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        logits = logits.float()
        targets = targets.float()

        # BCE Loss
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='mean')
        bce = bce_loss_fn(logits, targets)

        # Dice Loss
        topo_loss=0
        # print("logits.shape, targets.shape", logits.shape, targets.shape)
        for i in range(logits.shape[0]):
            topo_loss += getTopoLoss(logits[i], targets[i])
        topo_loss /= logits.shape[0]

        return bce + self.lambda_ * topo_loss
    
# -------------------------
# Iterative RRLoss
# -------------------------

import torch.nn as nn
import torch
import torchvision.utils as vutils

class BCE3Loss(nn.Module):
    """ BCE3 loss for the simultaneous segmentation of arteries [A], veins [V]
    and vessel tree [VT] (AV3).
    indices:
        artery: 0
        vein: 1
        vessel_tree (artery+vein): 2
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_vessels, vessels, only_av=False):
        if pred_vessels.ndim == 3:
            pred_vessels = pred_vessels.unsqueeze(0)

        if vessels.ndim == 3:
            vessels = vessels.unsqueeze(0)

        C = pred_vessels.shape[1]

        if C == 1:
            return self.loss(pred_vessels, vessels)

        elif C == 2:
            pred_a, pred_v = pred_vessels[:, 0], pred_vessels[:, 1]
            gt_a, gt_v = vessels[:, 0], vessels[:, 1]
            
            return self.loss(pred_a, gt_a) + self.loss(pred_v, gt_v)

        elif C >= 3:
            pred_a, pred_v, pred_vt = pred_vessels[:, 0], pred_vessels[:, 1], pred_vessels[:, 2]
            gt_a, gt_v, gt_vt = vessels[:, 0], vessels[:, 1], vessels[:, 2]

            loss = self.loss(pred_a, gt_a) + self.loss(pred_v, gt_v)

            if not only_av:
                loss += self.loss(pred_vt, gt_vt)

            return loss

    def save_predicted(self, prediction, fname):
        prediction_processed = self.process_predicted(prediction)
        vutils.save_image(prediction_processed, fname)

    def process_predicted(self, prediction):
        return torch.sigmoid(prediction.clone())

class RRLoss(nn.Module):
    """Recursive refinement loss.
    """
    def __init__(self, base_criterion, refine_only_av=True):
        super().__init__()
        self.base_criterion = base_criterion
        self.refine_only_av=refine_only_av

    def forward(self, predictions, gt):
        v_channel = gt[:, 0:1, :, :] | gt[:, 1:2, :, :]  # shape (2, 1, 512, 512)
        gt = torch.cat([gt, v_channel], dim=1) 
        
        gt = gt.float()

        loss_1 = self.base_criterion(predictions[0], gt)
        if len(predictions) == 1:
            return loss_1

        # mask = torch.sigmoid(predictions[0][:,2,:,:])

        # Second loss (refinement) inspired by Mosinska:CVPR:2018.
        loss_2 = 1 * self.base_criterion(predictions[1], gt, only_av=self.refine_only_av)
        if len(predictions) == 2:
            return loss_1 + loss_2
        for i, prediction in enumerate(predictions[2:], 2):
            loss_2 += i * self.base_criterion(prediction, gt, only_av=self.refine_only_av)

        K = len(predictions[1:])
        Z = (1/2) * K * (K + 1)

        loss_2 *= 1/Z

        loss = loss_1 + loss_2

        return loss

    def save_predicted(self, predictions, fname):
        self.base_criterion.save_predicted(predictions[-1], fname)

    def process_predicted(self, predictions):
        new_predictions = []
        for prediction in predictions:
            new_predictions.append(self.base_criterion.process_predicted(prediction))
        return new_predictions

class IterativeLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, preds, target):
        target = (target > 0).float()
        if not isinstance(preds, list):
            return self.base_loss(preds, target)

        total = 0
        K = len(preds)

        for i, p in enumerate(preds):
            weight = (i+1)
            total += weight * self.base_loss(p, target)

        return total / (K*(K+1)/2)