import torch


def compute_iou(bbox_1: torch.Tensor, bbox_2: torch.Tensor) -> torch.Tensor:
    
    assert bbox_1.shape == bbox_2.shape

    ########################################################################
    # TODO:                                                                #
    # Compute the intersection over union (IoU) for two batches of         #
    # bounding boxes, each of shape (B, 4). The result should be a tensor  #
    # of shape (B,).                                                       #
    # NOTE: the format of the bounding boxes is (ltrb), meaning            #
    # (left edge, top edge, right edge, bottom edge). Remember the         #
    # orientation of the image coordinates.                                #
    # NOTE: First calculate the intersection and use this to compute the   #
    # union                                                                #
    # iou = ...                                                            #
    ########################################################################

    # Unpack bounding box coordinates
    # print("bbox_1",bbox_1)
    # print("bbox_2",bbox_2)

    # print("bbox_1.shape",bbox_1.shape)
    left = torch.max(bbox_1[:, 0], bbox_2[:, 0])
    # print("left",left)
    # print("left.shape",left.shape)
    top = torch.max(bbox_1[:, 1], bbox_2[:, 1])
    right = torch.min(bbox_1[:, 2], bbox_2[:, 2])
    bottom = torch.min(bbox_1[:, 3], bbox_2[:, 3])

    # Calculate intersection area
    intersection = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)
    # print("intersection",intersection)
    # Calculate areas of each bounding box
    area_bbox1 = (bbox_1[:, 2] - bbox_1[:, 0]) * (bbox_1[:, 3] - bbox_1[:, 1])
    area_bbox2 = (bbox_2[:, 2] - bbox_2[:, 0]) * (bbox_2[:, 3] - bbox_2[:, 1])

    # Calculate union area
    union = area_bbox1 + area_bbox2 - intersection

    # Calculate IoU
    iou = intersection / union
    # print(iou.shape)
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return iou
