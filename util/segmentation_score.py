import torch
import numpy as np
from typing import Union, List, Tuple

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """
    
    def __init__(self, nclass, choice: str, filter_bg: bool) -> None:
        """_summary_

        Args:
            nclass (int): The number of classes.
            choice (str): ['full', 'upper_lower', 'upper']
        """
        super().__init__()
        self.nclass = nclass
        self.choice = choice
        self.filter_bg = filter_bg
        self.reset()
        
    def update(self, preds: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]], labels: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]):
        """Updates the internal evaluation result.
        
        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        
        def evaluate_worker(self, pred, label):
            if self.choice == 'full':
                correct, labeled = batch_pix_accuracy(pred, label, self.filter_bg) # based on the level of pixels
                inter, union = batch_intersection_union(pred, label, self.nclass, self.filter_bg)
            elif self.choice == 'upper_lower' or self.choice == 'upper':
                correct, labeled = batch_pix_accuracy_upper_lower_clothes(pred, label, self.filter_bg, self.choice) # based on the level of pixels
                inter, union = batch_intersection_union_upper_lower_clothes(pred, label, self.nclass, self.filter_bg, self.choice)
            
            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union
            
            return correct, labeled, inter, union
        
        if isinstance(preds, torch.Tensor):
            self.current_correct, self.current_label, self.current_inter, self.current_union = evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for index, (pred, label) in enumerate(zip(preds, labels)):
                if index == 0:
                    self.current_correct, self.current_label, self.current_inter, self.current_union = evaluate_worker(self, pred, label)
                else:
                    evaluate_worker(self, pred, label)
                
    def get(self) -> Tuple[float, float]:
        """Gets the all evaluation results for test sets.
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU
    
    def get_current(self) -> Tuple[float, float]:
        """Gets the current evaluation results for each step

        Returns:
            Tuple[float, float]: pixAcc, mIoU for each step
        """
        current_pixAcc = 1.0 * self.current_correct / (2.220446049250313e-16 + self.current_label)  # remove np.spacing(1)
        current_IoU = 1.0 * self.current_inter / (2.220446049250313e-16 + self.current_union)
        current_mIoU = current_IoU.mean().item()
        return current_pixAcc, current_mIoU
        
    def reset(self):
        """Resets the internal evaluation result to initial state.
        """
        if self.choice == 'full':
            self.total_inter = torch.zeros(self.nclass)
            self.total_union = torch.zeros(self.nclass)
            
            self.current_inter = torch.zeros(self.nclass) # current_step
            self.current_union = torch.zeros(self.nclass)
            
        elif self.choice == 'upper_lower':
            self.total_inter = torch.zeros(2)
            self.total_union = torch.zeros(2)
            
            self.current_inter = torch.zeros(2)
            self.current_union = torch.zeros(2)
        elif self.choice == 'upper':
            self.total_inter = torch.zeros(1)
            self.total_union = torch.zeros(1)
            
            self.current_inter = torch.zeros(1)
            self.current_union = torch.zeros(1)
        
        self.total_correct = 0
        self.total_label = 0
        
        # current step 
        self.current_correct = 0
        self.current_label = 0
        
# pytorch version
def batch_pix_accuracy(output: torch.Tensor, target: torch.Tensor, filter_bg: bool):
    """PixAcc

    Args:
        output (_type_): 4D Tensor
        target (_type_): 3D Tensor
    """
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1 # why + 1
    
    pixel_labeled = torch.sum(target > 0).item() if not filter_bg else torch.sum(target > 1).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item() if not filter_bg else torch.sum((predict == target) * (target > 1)).item() # predict > 0
    assert pixel_correct <= pixel_labeled, "Correct are should be smaller than label"

    return pixel_correct, pixel_labeled


def batch_pix_accuracy_upper_lower_clothes(output: torch.Tensor, target: torch.Tensor, filter_bg: bool, upper_lower_choice: str):
    """Cal the pixel accuracy for upper and lower clothes, upper clothes.

    Args:
        output (torch.Tensor): 4D Tensor
        target (torch.Tensor): 3D Tensor
        upper_lower_choice(str): 'upper_lower', 'upper'
    """
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1 # why + 1
    
    if upper_lower_choice == 'upper_lower':
        part_indexes = [4, 6]
    elif upper_lower_choice == 'upper':
        part_indexes = [4]
        
    
    pixel_labeled = sum([torch.sum(target == index).item() for index in part_indexes])
    pixel_correct = sum([torch.sum((predict == target) * (target == index)).item() for index in part_indexes]) # predict > 0
    assert pixel_correct <= pixel_labeled, "Correct are should be smaller than label"

    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass, filter_bg: bool):
    """mIoU

    Args:
        output (_type_): 4D Tensor
        target (_type_): 3D Tensor
        nclass (int): The number of classes
    """
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    
    # import pdb; pdb.set_trace()
    
    predict = predict.float() * (target > 0).float() if not filter_bg else predict.float() * (target > 1).float()
    intersection = predict * (predict == target).float()
    # choose the values of related indexes
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins= nbins, min= mini, max= maxi)
    area_lab = torch.histc(target.cpu(), bins= nbins, min=mini, max=maxi) if not filter_bg else torch.histc(target.cpu()[target> 1], bins= nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def batch_intersection_union_upper_lower_clothes(output, target, nclass, filter_bg: bool, upper_lower_choice: str):
    """mIoU

    Args:
        output (_type_): 4D Tensor
        target (_type_): 3D Tensor
        nclass (int): The number of classes
        upper_lower_choice(str): 'upper_lower', 'upper'
    """
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    
    if upper_lower_choice == 'upper_lower':
        part_indexes = [3, 5]
    elif upper_lower_choice == 'upper':
        part_indexes = [3]
    
    # import pdb; pdb.set_trace()
    
    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # choose the values of related indexes
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)[part_indexes]
    area_pred = torch.histc(predict.cpu(), bins= nbins, min= mini, max= maxi)[part_indexes]
    area_lab = torch.histc(target.cpu(), bins= nbins, min=mini, max=maxi)[part_indexes]
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()