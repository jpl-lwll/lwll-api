# Copyright (c) 2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from collections import defaultdict

__all__ = ['mean_average_precision', 'format_obj_detection_data', 'mAP_ranges']

"""
This code was largely taken from: https://github.com/Cartucho/mAP and modified
in order to work in memory and not use temporary files saving to disk throughout the process.

The initial pass through this code was to remove a lot of unecessary functionality related to plotting
as well as trying to group logical pieces of initial script. I think the person who initially wrote this
is a Matlap person hence why the code style is not consistent with the rest of this code base. - MH 12/15/19
"""


def log_average_miss_rate(precision: np.array, fp_cumsum: np.array, num_images: int) -> Tuple[float, float, float]:
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0.0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def voc_ap(rec: list, prec: list) -> Tuple[float, list, list]:
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    # This part makes the precision monotonically decreasing

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def populate_gt_stats(ground_truth_labels: list) -> Tuple[dict, dict, dict, list, int]:

    out_dict = {}

    gt_counter_per_class: defaultdict = defaultdict(int)
    counter_images_per_class: defaultdict = defaultdict(int)

    unique_images = sorted(set([line[0] for line in ground_truth_labels]))
    for file_id in unique_images:
        lines_list = [
            line for line in ground_truth_labels if line[0] == file_id]

        # create ground-truth dictionary
        bounding_boxes = []
        already_seen_classes: list = []
        for line in lines_list:
            class_name, left, top, right, bottom = line[1], line[2], line[3], line[4], line[5]
            bbox = f"{left} {top} {right} {bottom}"
            bounding_boxes.append(
                {"class_name": class_name, "bbox": bbox})
            # count that object
            gt_counter_per_class[class_name] += 1

            if class_name not in already_seen_classes:
                counter_images_per_class[class_name] += 1
                already_seen_classes.append(class_name)

        out_dict[file_id] = bounding_boxes

        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)

    return out_dict, gt_counter_per_class, counter_images_per_class, gt_classes, n_classes


def prep_preds(preds_list: list, gt_classes: list) -> dict:

    out_preds: dict = {class_name: [] for class_name in gt_classes}
    for line in preds_list:
        file_id, class_name, confidence, left, top, right, bottom = \
            line[0], line[1], line[2], line[3], line[4], line[5], line[6]

        if class_name in gt_classes:
            out_preds[class_name].append({
                "confidence": confidence,
                "file_id": file_id,
                "bbox": f"{left} {top} {right} {bottom}",
            })

    for class_name in out_preds.keys():
        out_preds[class_name].sort(key=lambda x: float(x['confidence']), reverse=True)

    return out_preds


def _add_used_var_to_bboxes(out_dict: dict) -> dict:
    _to_return = {}
    for key, val in out_dict.items():
        new_list = []
        for row in val:
            row['used'] = False
            new_list.append(row)
        _to_return[key] = new_list
    return _to_return


def calculate_mAP(out_dict: dict, out_preds: dict, gt_classes: list, counter_images_per_class: dict,
                  gt_counter_per_class: dict, n_classes: int, min_overlap: float = 0.5) -> Tuple[float, dict]:
    """
    Calculate the AP for each class
    """
    out_dict = _add_used_var_to_bboxes(out_dict)
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the results
    # print("# AP and precision/recall per class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
            Load detection-results of that class
        """
        dr_data = out_preds[class_name]

        """
            Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            # assign detection-results to ground truth object if any
            ground_truth_data = out_dict[file_id]
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                          min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                            (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        # print(f"Overlap: {ov}")
                        if ov > ovmax:
                            ovmax = ov  # type: ignore
                            gt_match = obj

            # assign detection as true positive/don't care/false positive
            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):  # type: ignore
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True  # type: ignore
                    count_true_positives[class_name] += 1
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    # status = "INSUFFICIENT OVERLAP"
                    pass

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])  # type: ignore

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(
            np.array(rec), np.array(fp), n_images)
        lamr_dictionary[class_name] = lamr

    mAP = round(sum_AP / n_classes, 2)
    return mAP, ap_dictionary


def mean_average_precision(ground_truth: list, predictions: list, overlap: float = 0.5) -> float:
    """

    :param ground_truth:
    :param predictions:
    :param overlap:
    :param classes: list of classes from DatasetMetadata
    :return:
    """
    out_dict, gt_counter_per_class, counter_images_per_class, gt_classes, n_classes = populate_gt_stats(
        ground_truth)
    out_preds = prep_preds(predictions, gt_classes)
    mAP, _ = calculate_mAP(out_dict, out_preds, gt_classes, counter_images_per_class,
                           gt_counter_per_class, n_classes, min_overlap=overlap)

    return mAP


def mAP_ranges(ground_truth: list, predictions: list, min_overlap: float = 0.5, max_overlap: float = 0.5, step: float = 0.05) -> Dict:
    """

    :param ground_truth:
    :param predictions:
    :param overlap:
    :param classes: list of classes from DatasetMetadata
    :return:
    """
    out_dict, gt_counter_per_class, counter_images_per_class, gt_classes, n_classes = populate_gt_stats(
        ground_truth)
    out_preds = prep_preds(predictions, gt_classes)

    overlap_levels = {}
    for overlap in np.arange(min_overlap, max_overlap, step):
        overlap = float(np.round(overlap, 2))
        mAP, _ = calculate_mAP(
            out_dict, out_preds, gt_classes, counter_images_per_class, gt_counter_per_class, n_classes, min_overlap=overlap)
        overlap_levels[overlap] = mAP
    return overlap_levels


def format_obj_detection_data(inp: pd.DataFrame) -> list:
    """
    Takes our DataFrame format input and converts to the appropriate list of lists format
    """
    new = inp['bbox'].str.split(",", n=4, expand=True)
    new.columns = ['xmin', 'ymin', 'xmax', 'ymax']
    inp = inp.merge(new, left_index=True, right_index=True)

    if 'confidence' in inp.columns:
        inp = inp[['id', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']]
    else:
        inp = inp[['id', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]

    inp.loc[:, 'xmin'] = inp['xmin'].apply(lambda x: int(float(x)))
    inp.loc[:, 'ymin'] = inp['ymin'].apply(lambda x: int(float(x)))
    inp.loc[:, 'xmax'] = inp['xmax'].apply(lambda x: int(float(x)))
    inp.loc[:, 'ymax'] = inp['ymax'].apply(lambda x: int(float(x)))
    outp: list = inp.to_numpy().tolist()
    return outp
