from .occ_types import *
import torch.nn.functional as nnf

def occupancy_bce(predict: T, winding_gt: T, ignore: Optional[T] = None, *args) -> T:
    """
    Compute the Binary Cross-Entropy (BCE) loss for occupancy prediction.

    The function calculates the BCE loss between the predicted occupancy values 
    and the ground truth winding numbers. It optionally takes into account an 
    'ignore' tensor to mask certain elements from the loss computation.

    Parameters:
    - predict (T): A tensor containing the predicted occupancy values.
    - winding_gt (T): A tensor containing the ground truth winding numbers.
    - ignore (Optional[T]): An optional tensor that indicates elements to be 
      ignored in the loss computation. If provided, the loss for these elements 
      will be zero.

    Returns:
    - T: The computed BCE loss as a tensor.

    Note:
    - The function assumes that both 'predict' and 'winding_gt' tensors are 
      of the same shape.
    - If 'winding_gt' is not a boolean tensor, it is thresholded to create 
      a boolean mask where values greater than zero are considered as True.
    """

    if winding_gt.dtype is not torch.bool:
        winding_gt = winding_gt.gt(0)
    labels = winding_gt.flatten().float()
    predict = predict.flatten()
    if ignore is not None:
        ignore = (~ignore).flatten().float()
    loss = nnf.binary_cross_entropy_with_logits(predict, labels, weight=ignore)
    return loss


def reg_z_loss(z: T) -> T:
    """
    Compute the regularization loss based on the L2 norm of a tensor.

    This function calculates the mean L2 norm (Euclidean norm) of a given 
    tensor 'z', which can be used as a regularization loss in various 
    optimization problems.

    Parameters:
    - z (T): A tensor for which the regularization loss is to be computed.

    Returns:
    - T: The computed mean L2 norm as a tensor, representing the regularization loss.

    Note:
    - The function computes the L2 norm along the second dimension (axis=1) of the tensor 'z'.
    - The mean of these norms is then returned as the regularization loss.
    """
    norms = z.norm(2, 1)
    loss = norms.mean()
    return loss