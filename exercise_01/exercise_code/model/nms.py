import torch

from exercise_code.model.compute_iou import compute_iou


def non_maximum_suppression(bboxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    ########################################################################
    # TODO:                                                                #
    # Compute the non maximum suppression                                  #
    # Input:                                                               #
    # bounding boxes of shape B,4                                          #
    # scores of shape B                                                    #
    # threshold for iou: if the overlap is bigger, only keep one of the    #
    # bboxes                                                               #
    # Output:                                                              #
    # bounding boxes of shape B_,4                                         #
    ########################################################################
    # # Sort the bounding boxes based on scores in descending order
    # sorted_indices = torch.argsort(scores, descending=True)
    # bboxes = bboxes[sorted_indices]
    # scores = scores[sorted_indices]

    # selected_indices = []

    # while bboxes.shape[0] > 0:
    #     # Pick the bounding box with the highest score
    #     selected_bbox = bboxes[0]
    #     selected_indices.append(selected_bbox)
    #     # print("selected_bbox.unsqueeze(0)",selected_bbox.unsqueeze(0).shape)
    #     # print("bboxes_sorted[1:]",bboxes_sorted[1:].shape)
    #     # Compute IoU with the selected bounding box
    #     iou=[]
    #     for i in range(bboxes[1:,].shape[0]):
    #       if i != 0:
    #         iou.append(compute_iou(selected_bbox.unsqueeze(0), bboxes[i].unsqueeze(0)))

    #     iou=torch.tensor(iou)
    #     # print("iou.shape",iou.shape)
    #     # Remove bounding boxes with IoU greater than the threshold
    #     filtered_indices = torch.where(iou <= threshold)[0]
    #     # print("filtered_indices.shape",filtered_indices.shape)

    #     bboxes_sorted = bboxes[1:][filtered_indices]
    #     scores_sorted = scores[1:][filtered_indices]

    # # Convert the selected indices to a tensor
    # bboxes_nms = torch.stack(selected_indices)
    
    # Sort bounding boxes by their scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    bboxes = bboxes[sorted_indices]
    scores = scores[sorted_indices]

    keep = []
    while bboxes.shape[0] > 0:
        # Pick the box with the highest score
        current_box = bboxes[0]
        keep.append(current_box)

        # Compute IoU between the current box and all other boxes
        ious = compute_iou(current_box.unsqueeze(0).repeat(len(bboxes) - 1, 1), bboxes[1:])

        # Keep boxes where IoU is less than the threshold
        mask = ious <= threshold
        bboxes = bboxes[1:][mask]
        scores = scores[1:][mask]

    bboxes_nms = torch.stack(keep)


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return bboxes_nms
