import torch
import numpy as np


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def test_val_shared_step(x, y, prediction, seg_label_to_cat, seg_class_map, num_seg_classes):
    cur_batch_size, _, NUM_POINT = x.size()
    cur_pred_val = prediction.cpu().data.numpy()
    cur_pred_val_logits = cur_pred_val
    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
    target = y.cpu().data.numpy()

    # loss = self.loss_criterion(prediction.contiguous().view(cur_batch_size, -1, self.num_seg_classes), target)
    # self.log('val_loss', loss, on_step=True, on_epoch=False)

    for i in range(cur_batch_size):
        cat = seg_label_to_cat[target[i, 0]]
        logits = cur_pred_val_logits[i, :, :]
        cur_pred_val[i, :] = np.argmax(logits[:, seg_class_map[cat]], 1) + seg_class_map[cat][0]

    correct = np.sum(cur_pred_val == target)
    total_correct = correct
    total_seen = (cur_batch_size * NUM_POINT)

    total_seen_class = np.zeros(num_seg_classes)
    total_correct_class = np.zeros(num_seg_classes)

    for l in range(num_seg_classes):
        total_seen_class[l] = np.sum(target == l)
        total_correct_class[l] = (np.sum((cur_pred_val == l) & (target == l)))

    shape_ious = {cat: [] for cat in seg_class_map.keys()}
    for i in range(cur_batch_size):
        segp = cur_pred_val[i, :]
        segl = target[i, :]
        cat = seg_label_to_cat[segl[0]]
        part_ious = np.zeros(len(seg_class_map[cat]))
        for l in seg_class_map[cat]:
            if (np.sum(segl == l) == 0) and (
                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                part_ious[l - seg_class_map[cat][0]] = 1.0
            else:
                part_ious[l - seg_class_map[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    np.sum((segl == l) | (segp == l)))
        shape_ious[cat].append(np.mean(part_ious))

    # self.log('instance_avg_iou', iou, on_step=False, on_epoch=True, sync_dist=True)
    return {'total_correct': total_correct,
            'total_seen': total_seen, 'total_seen_class': total_seen_class,
            'total_correct_class': total_correct_class, 'shape_ious': shape_ious}


def test_val_shared_epoch(outputs, seg_class_map, num_seg_classes):
    total_correct = 0
    total_seen = 0
    total_seen_class = np.zeros(num_seg_classes)
    total_correct_class = np.zeros(num_seg_classes)
    shape_ious = {cat: [] for cat in seg_class_map.keys()}

    for output in outputs:
        total_correct += output['total_correct']
        total_seen += output['total_seen']
        total_seen_class += output['total_seen_class']
        total_correct_class += output['total_correct_class']

        for cat, values in output['shape_ious'].items():
            shape_ious[cat] += values

    all_shape_ious = []

    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_shape_ious = np.mean(list(shape_ious.values()))
    return shape_ious, \
        total_correct / float(total_seen),\
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)),\
        mean_shape_ious,\
        np.mean(all_shape_ious)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)