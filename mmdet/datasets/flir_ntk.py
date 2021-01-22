import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict
import json
from typing import Union, List

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset



@DATASETS.register_module()
class FlirNtk(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.ann_file_path = ann_file
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    try:
                        data['category_id'] = self.cat_ids[label]
                    except:
                        print(self.cat_ids)
                        print(label)
                        raise Exception("ERROR!")
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results: list,
                 metric: str = 'bbox',
                 label_vector: str = None,  # undocumented tmp arg
                 image_source_dir: Union[str, None] = None,  # undocumented tmp arg
                 image_prefix_dir: str = 'data',  # undocumented tmp arg
                 logger: logging.Logger = None,
                 jsonfile_prefix: Union[str, None] = None,
                 min_iou_threshold: float = 0.5,
                 classes_of_interest: Union[List[Union[str]], None] = None,
                 confidence_thresholds_of_interest: Union[List[float], None] = None,
                 recall_thresholds: Union[List[float], None] = None,
                 precision_thresholds: Union[List[float], None] = None,
                 metric_items: Union[List[str], None] = None) -> OrderedDict:
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Only support metric currently is 'bbox'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            min_iou_threshold (float), optional): IoU threshold used for
                evaluating recalls/mAPs. Default: 0.5.
            classes_of_interest (list[str]): the classes to evaluate on, must be class names by string. Default is just
            'all'
            confidence_thresholds_of_interest (list[float]): if using 'scores at confidence thresholds', the confidence
            thresholds to evaluate on. Default is 0.9
            recall_thresholds (list[float]): if using scores_at_recall_thresholds, the recall thresholds to evaluate at
            default is 0.9
            precision_thresholds (list[float]): if using scores_at_precision_thresholds, the precision thresholds to
            evaluate at. Default is 0.9.
            metric_items (list[str],  optional): Metric items that will
                be returned. Options are 'map_mar_fscore', 'scores_at_confidence_thresholds',
                'scores_at_recall_threshold', 'scores_at_precision_threshold' and 'pr_curve'. Will be used when
                `metric=='bbox' (currently only available option)` Default is map_mar_fscore and pr_curve.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        assure_python_path()
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        classes_of_interest = ['all'] if classes_of_interest is None else classes_of_interest
        confidence_thresholds_of_interest = [0.9] if confidence_thresholds_of_interest is None else \
            confidence_thresholds_of_interest
        recall_thresholds = [0.9] if recall_thresholds is None else recall_thresholds
        precision_thresholds = [0.9] if precision_thresholds is None else precision_thresholds
        metric_items = ['map_mar_fscore', 'pr_curve'] if metric_items is None else metric_items
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        # temp for testing
        image_source_dir = osp.basename(result_files[metric]) if image_source_dir is None else image_source_dir
        eval_results = OrderedDict()
        val_remap, detection_remap = None, None
        ntk_tensorboard_accumulator = NTKTensorboardAccumulator(
            val_ucoco_path=self.ann_file_path,  # since only bbox is supported
            label_file=label_vector,
            image_source_dir=image_source_dir,
            img_prefix_dir=image_prefix_dir,
            formatted_det_path=result_files[metric],
            val_remap=val_remap,
            detection_remap=detection_remap,
            min_iou_thr=min_iou_threshold)

        for class_of_interest in classes_of_interest:
            for metric_item in metric_items:
                if metric_item.lower() == 'map_mar_fscore':
                    eval_results = ntk_tensorboard_accumulator.write_map(class_of_interest, eval_results)
                elif metric_item.lower() == 'scores_at_confidence_threshold':
                    for confidence_threshold_of_interest in confidence_thresholds_of_interest:
                        eval_results = ntk_tensorboard_accumulator.write_scores_at_confidence_threshold(
                            class_of_interest, confidence_threshold_of_interest, eval_results)
                elif metric_item.lower() == 'scores_at_recall_threshold':
                    for recall_threshold in recall_thresholds:
                        eval_results = ntk_tensorboard_accumulator.write_scores_at_recall_threshold(
                            class_of_interest, recall_threshold, eval_results)
                elif metric_item.lower() == 'scores_at_precision_threshold':
                    for precision_threshold in precision_thresholds:
                        eval_results = ntk_tensorboard_accumulator.write_scores_at_precision_threshold(
                            class_of_interest, precision_threshold, eval_results)
                elif metric_item.lower() == 'pr_curve':
                    eval_results = ntk_tensorboard_accumulator.write_pr_curve(class_of_interest, eval_results)
                else:
                    raise Exception(f"Error! Unknown metric item {metric_item} available metric items are:"
                                    "'map_mar_fscore', 'scores_at_confidence_thresholds', 'scores_at_recall_threshold',"
                                    "'scores_at_precision_threshold' and 'pr_curve'")
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


def assure_python_path():
    try:
        from nneat.lib.load_data.load_ucoco_to_gt import compute as load_ucoco_gt
    except ModuleNotFoundError:
        raise Exception("Error! You must add both the ROOTDIR for neural-toolkit and " +
                        "${neural-toolkit-root-dir}/dataset-toolkit" +
                        "to your PYTHONPATH before using NTK based tensorboard metrics")


class NTKTensorboardAccumulator:
    def __init__(self, val_ucoco_path, label_file: str, image_source_dir: str, img_prefix_dir: str,
                 formatted_det_path: str, val_remap: str = None, detection_remap: str = None, min_iou_thr: float = 0.5,
                 ):
        """
        :param val_ucoco_path: Full path to the validation file in the ucoco json format
        :param label_file: full path to the label vector file
        :param image_source_dir: path to the images root
        :param img_prefix_dir: the prefix to the images dir
        :param formatted_det_path: the full path to the formatted detection json file,
        :param val_remap: a file for remapping the classes of the valdiation gt file
        :param detection_remap: a file for remapping the classes of the detection file
        """
        try:
            from nneat.lib.load_data.load_ucoco_to_gt import compute as load_ucoco_gt
            from nneat.lib.load_data.load_formatted_detections_to_detections import compute as load_fmt_det
            from nneat.lib.object_detector_metrics.cocoeval.run_cocoeval import compute as run_cocoeval
        except ModuleNotFoundError:
            raise Exception("Error! You must add both the ROOTDIR for nueral-toolkit and $ROOTDIR/dataset-toolkit" +\
                            "to your python path before using NTK based tensorboard metrics")
        val_remap = val_remap if val_remap != 'none' else None
        detection_remap = detection_remap if detection_remap != 'none' else None
        img_prefix_dir = img_prefix_dir if img_prefix_dir != 'none' else None
        gt_dict = load_ucoco_gt(val_ucoco_path, label_file, image_source_dir=image_source_dir,
                                prefix_dir=img_prefix_dir, remap=val_remap)
        gt, gt_im_store = gt_dict['gt'], gt_dict['gt_image_store']
        det = load_fmt_det(formatted_det_path, image_id_map=gt_dict['image_id_map'], label_vector=label_file,
                           remap=detection_remap)['detections']
        with open(label_file, 'r+') as lf:
            label_vector = json.load(lf)
        self.coco_eval = run_cocoeval(gt, det, label_vector, gt_im_store, min_iou_thr=min_iou_thr)

    def write_map(self, class_of_interest: Union[str, int], eval_results: OrderedDict) -> OrderedDict:
        """
        Adds mAP/mAR/Fscore to a tensorboard writer object using NTK
        :param class_of_interest: Class of interest to calculate scores on
        :param eval_results: the dictionary log
        :return: the altered eval results
        """
        from nneat.lib.object_detector_metrics.cocoeval.score_overall_performance import compute as \
            score_overall_performance
        sop_result = score_overall_performance(self.coco_eval, class_of_interest)
        size_list = ['all', 'small', 'medium', 'large']
        metric_list = ['mAP', 'mAR', 'Fscore']
        metric_size_product = itertools.product(size_list, metric_list)
        for size, metric in metric_size_product:
            metric_name = f"bbox_{metric}_class_{class_of_interest}_size_{size}"
            metric_score = sop_result.get_score(score_name=metric, size=size)
            eval_results[metric_name] = metric_score
        return eval_results

    def write_scores_at_confidence_threshold(self,
                                             class_of_interest: Union[str, int],
                                             confidence_threshold: float,
                                             eval_results: OrderedDict) -> OrderedDict:
        """
        Use NTK to get the number of TPs, FPs, FNs, on the gt at a confidence score and writes it to a tensorboard
        object
        :param class_of_interest: Class of interest to calculate scores on
        :param confidence_threshold: confidence threshold to evaluate at
        :param eval_results: the dictionary log
        :return: the altered eval results
        """
        from nneat.lib.object_detector_metrics.cocoeval.score_operating_point import compute as \
            score_operating_point
        sop_result = score_operating_point(cocoeval_data=self.coco_eval,
                                           confidence_threshold=confidence_threshold,
                                           confidence_name="None",
                                           cat_id=class_of_interest)
        size_list = ['all', 'small', 'medium', 'large']
        metric_list = ['tp', 'fn', 'fp']
        metric_size_product = itertools.product(size_list, metric_list)
        for size, metric in metric_size_product:
            metric_name = f"bbox_{metric}_class_{class_of_interest}_size_{size}"
            metric_score = sop_result.get_metric(size_class=size, metric=metric)
            eval_results[metric_name] = metric_score
        return eval_results

    def write_scores_at_recall_threshold(self,
                                         class_of_interest: Union[str, int],
                                         recall_threshold: float,
                                         eval_results: OrderedDict):
        """
        Use NTK to get the number of TPs, FPs, FNs, on the gt at a recall threshold and writes it to a tensorboard
        object
        :param class_of_interest: Class of interest to calculate scores on
        :param recall_threshold: recall threshold to evaluate at
        :param eval_results: the dictionary log
        :return: the altered eval results
        """
        from nneat.lib.object_detector_metrics.cocoeval.set_operating_point_and_score import compute as \
            set_operating_point_and_score

        sopas_result = set_operating_point_and_score(cocoeval_data=self.coco_eval,
                                                     cat_id=class_of_interest,
                                                     desired_recall=recall_threshold,
                                                     find_best_if_fail=True)
        actual_recall = sopas_result.get_actual_recall()
        eval_results[f"desired_recall_{str(recall_threshold)}_actual_recall_class{class_of_interest}"] = actual_recall
        actual_precision = sopas_result.get_actual_precision()
        eval_results[f"desired_recall_{str(recall_threshold)}_actual_precision_class{class_of_interest}"] =\
            actual_precision
        return eval_results

    def write_scores_at_precision_threshold(self,
                                            class_of_interest: Union[str, int],
                                            precision_threshold: float,
                                            eval_results: OrderedDict):
        """
        Use NTK to get the number of TPs, FPs, FNs, on the gt at a recall threshold and writes it to a tensorboard
        object
        :param class_of_interest: Class of interest to calculate scores on
        :param precision_threshold: recall threshold to evaluate at
        :param eval_results: the dictionary log
        :return: the altered eval results
        """
        from nneat.lib.object_detector_metrics.cocoeval.set_operating_point_and_score import compute as \
            set_operating_point_and_score

        sopas_result = set_operating_point_and_score(cocoeval_data=self.coco_eval,
                                                     cat_id=class_of_interest,
                                                     desired_precision=precision_threshold,
                                                     find_best_if_fail=True)
        actual_recall = sopas_result.get_actual_recall()
        eval_results[f"desired_precision_{str(precision_threshold)}_actual_recall_class{class_of_interest}"] = \
            actual_recall
        actual_precision = sopas_result.get_actual_precision()
        eval_results[f"desired_precision_{str(precision_threshold)}_actual_precision_class{class_of_interest}"] =\
            actual_precision
        return eval_results

    def write_pr_curve(self, class_of_interest: Union[str, int], eval_results: OrderedDict):
        """
        Adds mAP/mAR/Fscore to a tensorboard writer object using NTK
        :param class_of_interest: Class of interest to calculate scores on
        :param eval_results: the dictionary log
        :return: the altered eval results
        """
        from nneat.lib.object_detector_metrics.cocoeval.plot_pr import compute as plot_pr
        pr_curve_name = f"PR_Curve_class_{class_of_interest}"
        plot_pr_result = plot_pr(cocoeval_data=self.coco_eval,
                                 cat_id=class_of_interest,
                                 size_class=[0, 1, 2, 3],
                                 line_names=["All", "Small", "Medium", "Large"],
                                 title=pr_curve_name)
        eval_results[f"bbox_{pr_curve_name}"] = plot_pr_result.get_figure()
        return eval_results

