import numpy as np
from collections import defaultdict
import pycocotools._mask as _mask
from terminaltables import AsciiTable
import matplotlib.pyplot as plt
import os
# import pdb

def replace_nan_with_zero(data):
    """
    递归地将字典或列表中的NaN值替换为0.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = replace_nan_with_zero(value)
    elif isinstance(data, list):
        return [replace_nan_with_zero(item) for item in data]
    elif isinstance(data, float) and np.isnan(data):
        return 0.0
    return data

class SelfEval:
    def __init__(self, cocoGt, cocoDt, all_points=False, iou_type='bbox', iou_thres=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        assert iou_type in ('bbox', 'segmentation'), 'Only support measure bbox or segmentation now.'
        self.NAMES = [cat['name'] for cat in cocoGt.cats.values()]
        self.iou_type = iou_type
        self.gt = defaultdict(list)
        self.dt = defaultdict(list)
        self.all_points = all_points

        # np.arange and np.linspace can not get the accurate number, e.g. 0.8500000000000003 and 0.8999999999
        self.iou_thre = iou_thres
        self.recall_points = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)

        self.max_det = 100
        self.area = [[0 ** 2, 1e5 ** 2]]
        self.area_name = ['all']

        self.imgIds = list(np.unique(cocoGt.getImgIds()))
        self.catIds = list(np.unique(cocoGt.getCatIds()))

        gts = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=self.imgIds, catIds=self.catIds))
        dts = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=self.imgIds, catIds=self.catIds))

        if iou_type == 'segmentation':
            for ann in gts:
                rle = cocoGt.annToRLE(ann)
                ann['segmentation'] = rle
            for ann in dts:
                rle = cocoDt.annToRLE(ann)
                ann['segmentation'] = rle

        self.C, self.A, self.T, self.N = len(self.catIds), len(self.area), len(self.iou_thre), len(self.imgIds)

        # key is a tuple (gt['image_id'], gt['category_id']), value is a list.
        for gt in gts:
            # if gt['iscrowd'] == 0:  # TODO: why this makes the result different
            self.gt[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self.dt[dt['image_id'], dt['category_id']].append(dt)

        print()
        print(f'---------------------Evaluating "{self.iou_type}"---------------------')

    def evaluate(self):
        self.match_record = [[['no_gt_no_dt' for _ in range(self.N)] for _ in range(self.A)] for _ in range(self.C)]

        for c, cat_id in enumerate(self.catIds):
            for a, area in enumerate(self.area):
                for n, img_id in enumerate(self.imgIds):
                    print(f'\rMatching ground-truths and detections: C: {c}, A: {a}, N: {n}', end='')

                    gt_list, dt_list = self.gt[img_id, cat_id], self.dt[img_id, cat_id]

                    if len(gt_list) == 0 and len(dt_list) == 0:
                        continue
                    elif len(gt_list) != 0 and len(dt_list) == 0:
                        for one_gt in gt_list:
                            if one_gt['iscrowd'] or one_gt['area'] < area[0] or one_gt['area'] > area[1]:
                                one_gt['_ignore'] = 1
                            else:
                                one_gt['_ignore'] = 0

                        # sort ignored gt to last
                        index = np.argsort([aa['_ignore'] for aa in gt_list], kind='mergesort')
                        gt_list = [gt_list[i] for i in index]

                        gt_ignore = np.array([aa['_ignore'] for aa in gt_list])
                        num_gt = np.count_nonzero(gt_ignore == 0)
                        self.match_record[c][a][n] = {'has_gt_no_dt': 'pass', 'num_gt': num_gt}
                    else:
                        # different sorting method generates slightly different results.
                        # 'mergesort' is used to be consistent as the COCO Matlab implementation.
                        index = np.argsort([-aa['score'] for aa in dt_list], kind='mergesort')
                        dt_list = [dt_list[i] for i in index]
                        dt_list = dt_list[0: self.max_det]  # if len(one_dt) < self.max_det, no influence

                        if len(gt_list) == 0 and len(dt_list) != 0:
                            dt_matched = np.zeros((self.T, len(dt_list)))  # all dt shoule be fp, so set as 0
                            # set unmatched detections which are outside of area range to ignore
                            dt_out_range = [aa['area'] < area[0] or aa['area'] > area[1] for aa in dt_list]
                            dt_ignore = np.repeat(np.array(dt_out_range)[None, :], repeats=self.T, axis=0)
                            num_gt = 0
                        else:
                            for one_gt in gt_list:
                                if one_gt['iscrowd'] or one_gt['area'] < area[0] or one_gt['area'] > area[1]:
                                    one_gt['_ignore'] = 1
                                else:
                                    one_gt['_ignore'] = 0

                            # sort ignored gt to last
                            index = np.argsort([aa['_ignore'] for aa in gt_list], kind='mergesort')
                            gt_list = [gt_list[i] for i in index]

                            gt_matched = np.zeros((self.T, len(gt_list)))
                            gt_ignore = np.array([aa['_ignore'] for aa in gt_list])
                            dt_matched = np.zeros((self.T, len(dt_list)))
                            dt_ignore = np.zeros((self.T, len(dt_list)))

                            box_gt = [aa[self.iou_type] for aa in gt_list]
                            box_dt = [aa[self.iou_type] for aa in dt_list]

                            iscrowd = [int(aa['iscrowd']) for aa in gt_list]
                            IoUs = _mask.iou(box_dt, box_gt, iscrowd)  # shape: (num_dt, num_gt)

                            assert len(IoUs) != 0, 'Bug, IoU should not be None when gt and dt are both not empty.'
                            for t, one_thre in enumerate(self.iou_thre):
                                for d, one_dt in enumerate(dt_list):
                                    iou = one_thre
                                    g_temp = -1
                                    for g in range(len(gt_list)):
                                        # if this gt already matched, and not a crowd, continue
                                        if gt_matched[t, g] > 0 and not iscrowd[g]:
                                            continue
                                        # if dt matched a ignore gt, break, because all the ignore gts are at last
                                        if g_temp > -1 and gt_ignore[g_temp] == 0 and gt_ignore[g] == 1:
                                            break
                                        # continue to next gt unless better match made
                                        if IoUs[d, g] < iou:
                                            continue
                                        # if match successful and best so far, store appropriately
                                        iou = IoUs[d, g]
                                        g_temp = g

                                    # if match made store id of match for both dt and gt
                                    if g_temp == -1:
                                        continue

                                    dt_ignore[t, d] = gt_ignore[g_temp]
                                    dt_matched[t, d] = gt_list[g_temp]['id']
                                    gt_matched[t, g_temp] = one_dt['id']

                            dt_out_range = [aa['area'] < area[0] or aa['area'] > area[1] for aa in dt_list]
                            dt_out_range = np.repeat(np.array(dt_out_range)[None, :], repeats=self.T, axis=0)
                            dt_out_range = np.logical_and(dt_matched == 0, dt_out_range)

                            dt_ignore = np.logical_or(dt_ignore, dt_out_range)
                            num_gt = np.count_nonzero(gt_ignore == 0)

                        self.match_record[c][a][n] = {'dt_match': dt_matched,
                                                      'dt_score': [aa['score'] for aa in dt_list],
                                                      'dt_ignore': dt_ignore,
                                                      'num_gt': num_gt}

    def accumulate(self):  # self.match_record is all this function need
        print('\nComputing recalls and precisions...')

        R = len(self.recall_points)

        self.p_record = [[[None for _ in range(self.T)] for _ in range(self.A)] for _ in range(self.C)]
        self.r_record = [[[None for _ in range(self.T)] for _ in range(self.A)] for _ in range(self.C)]
        self.s_record = [[[None for _ in range(self.T)] for _ in range(self.A)] for _ in range(self.C)]

        # TODO: check if the logic is right, especially when there are absent categories when evaling part of images
        for c in range(self.C):
            for a in range(self.A):
                temp_dets = self.match_record[c][a]
                temp_dets = [aa for aa in temp_dets if aa != 'no_gt_no_dt']

                num_gt = sum([aa['num_gt'] for aa in temp_dets])
                assert num_gt != 0, f'Error, category {self.NAMES[c]} does not exist in validation images.'

                # exclude images which have no dt
                temp_dets = [aa for aa in temp_dets if 'has_gt_no_dt' not in aa]

                if len(temp_dets) == 0:  # if no detection found for all validation images
                    # If continue directly, the realted record would be 'None',
                    # which is excluded when computing mAP in summarize().
                    for t in range(self.T):
                        self.p_record[c][a][t] = np.array([0.])
                        self.r_record[c][a][t] = np.array([0.])
                        self.s_record[c][a][t] = np.array([0.])
                    continue

                scores = np.concatenate([aa['dt_score'] for aa in temp_dets])
                index = np.argsort(-scores, kind='mergesort')
                score_sorted = scores[index]

                dt_matched = np.concatenate([aa['dt_match'] for aa in temp_dets], axis=1)[:, index]
                dt_ignore = np.concatenate([aa['dt_ignore'] for aa in temp_dets], axis=1)[:, index]

                tps = np.logical_and(dt_matched, np.logical_not(dt_ignore))  # shape: (thre_num, dt_num)
                fps = np.logical_and(np.logical_not(dt_matched), np.logical_not(dt_ignore))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype='float32')
                fp_sum = np.cumsum(fps, axis=1).astype(dtype='float32')

                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    recall = (tp / num_gt).tolist()
                    precision = (tp / (fp + tp + np.spacing(1))).tolist()

                    # numpy is slow without cython optimization for accessing elements
                    # use python list can get significant speed improvement
                    p_smooth = precision.copy()
                    for i in range(len(tp) - 1, 0, -1):
                        if p_smooth[i] > p_smooth[i - 1]:
                            p_smooth[i - 1] = p_smooth[i]

                    if self.all_points:
                        p_reduced, s_reduced = [], []
                        r_reduced = list(set(recall))
                        r_reduced.sort()

                        for one_r in r_reduced:
                            index = recall.index(one_r)  # the first precision w.r.t the recall is always the highest
                            p_reduced.append(p_smooth[index])
                            s_reduced.append(score_sorted[index])

                        stair_h, stair_w, stair_s = [], [], []
                        for i in range(len(p_reduced)):  # get the falling edge of the stairs
                            if (i != len(p_reduced) - 1) and (p_reduced[i] > p_reduced[i + 1]):
                                stair_h.append(p_reduced[i])
                                stair_w.append(r_reduced[i])
                                stair_s.append(s_reduced[i])

                        stair_h.append(p_reduced[-1])  # add the final point which is out of range in the above loop
                        stair_w.append(r_reduced[-1])
                        stair_s.append(s_reduced[-1])

                        stair_w.insert(0, 0.)  # insert 0. at index 0 to do np.diff()
                        stair_w = np.diff(stair_w)
                        self.p_record[c][a][t] = np.array(stair_h)
                        self.r_record[c][a][t] = np.array(stair_w)
                        self.s_record[c][a][t] = np.array(stair_s)
                    else:
                        index = np.searchsorted(recall, self.recall_points, side='left')
                        score_101, precision_101 = np.zeros((R,)), np.zeros((R,))
                        # if recall is < 1.0, then there will always be some points out of the recall range,
                        # so use try...except... to deal with it automatically.
                        try:
                            for ri, pi in enumerate(index):
                                precision_101[ri] = p_smooth[pi]
                                score_101[ri] = score_sorted[pi]
                        except:
                            pass

                        self.p_record[c][a][t] = precision_101
                        num_points = len(precision_101)
                        # COCO's ap = mean of the 101 precision points, I use this way to keep the code compatibility,
                        # so the width of the stair is 1 / num_points. This can get the same AP. But recall is
                        # different. COCO's recall is the last value of all recall values, and mine is the last value
                        # of 101 recall values.
                        self.r_record[c][a][t] = np.array([1 / num_points] * num_points)
                        self.s_record[c][a][t] = score_101

    @staticmethod
    def mr4(array):
        return round(float(np.mean(array)), 4)

    def summarize(self):
        print('Summarizing...')
        self.AP_matrix = np.zeros((self.C, self.A, self.T)) - 1
        self.AR_matrix = np.zeros((self.C, self.A, self.T)) - 1
        if self.all_points:
            self.MPP_matrix = np.zeros((self.C, self.A, self.T, 5)) - 1

        for c in range(self.C):
            for a in range(self.A):
                for t in range(self.T):
                    if self.p_record[c][a][t] is not None:  # exclude absent categories, the related AP is -1
                        self.AP_matrix[c, a, t] = (self.p_record[c][a][t] * self.r_record[c][a][t]).sum()
                        # In all points mode, recall is always the sum of 'stair_w', but in 101 points mode,
                        # we need to find where precision reduce to 0., and thus calculate the recall.
                        if self.all_points:
                            self.AR_matrix[c, a, t] = self.r_record[c][a][t].sum()
                            r_cumsum = np.cumsum(self.r_record[c][a][t])
                            ap_array = self.p_record[c][a][t] * r_cumsum
                            index = np.argmax(ap_array)
                            p_max = self.p_record[c][a][t][index]
                            r_max = r_cumsum[index]
                            s_max = self.s_record[c][a][t][index]
                            mpp = ap_array[index]
                            # If ap == 0 for a certain threshold, ff should be taken into calculation because
                            # it's not an absent category, so ff should be 0 instead of nan.
                            ff = 0. if self.AP_matrix[c, a, t] == 0 else mpp / self.AP_matrix[c, a, t]
                            self.MPP_matrix[c, a, t] = np.array([p_max, r_max, s_max, mpp, ff])
                        else:
                            r_mask = self.p_record[c][a][t] != 0
                            self.AR_matrix[c, a, t] = (self.r_record[c][a][t])[r_mask].sum()

        table_c_list = [['Category', 'AP', 'Recall'] * 3]
        c_line = ['all', self.mr4(self.AP_matrix[:, 0, :]), self.mr4(self.AR_matrix[:, 0, :])]

        if self.all_points:  # max practical precision
            table_mpp_list = [['Category', 'P_max', 'R_max', 'Score', 'MPP', 'FF'] * 3]
            mpp_line = ['all', self.mr4(self.MPP_matrix[:, 0, :, 0]), self.mr4(self.MPP_matrix[:, 0, :, 1]),
                        self.mr4(self.MPP_matrix[:, 0, :, 2]), self.mr4(self.MPP_matrix[:, 0, :, 3]),
                        self.mr4(self.MPP_matrix[:, 0, :, 4])]

        for i in range(self.C):
            if -1 in self.AP_matrix[i, 0, :]:  # if this category is absent
                assert self.AP_matrix[i, 0, :].sum() == -len(self.iou_thre), 'Not all ap is -1 in absent category'
                c_line += [self.NAMES[i], 'absent', 'absent']
                if self.all_points:
                    mpp_line += [self.NAMES[i], 'absent', 'absent', 'absent', 'absent', 'absent']
            else:
                c_line += [self.NAMES[i], self.mr4(self.AP_matrix[i, 0, :]), self.mr4(self.AR_matrix[i, 0, :])]
                if self.all_points:
                    mpp_line += [self.NAMES[i], self.mr4(self.MPP_matrix[i, 0, :, 0]),
                                 self.mr4(self.MPP_matrix[i, 0, :, 1]), self.mr4(self.MPP_matrix[i, 0, :, 2]),
                                 self.mr4(self.MPP_matrix[i, 0, :, 3]), self.mr4(self.MPP_matrix[i, 0, :, 4])]
            if (i + 2) % 3 == 0:
                table_c_list.append(c_line)
                c_line = []

                if self.all_points:
                    table_mpp_list.append(mpp_line)
                    mpp_line = []

        table_iou_list = [['IoU'] + self.iou_thre, ['AP'], ['Recall']]
        for i in range(self.T):
            ap_m = self.AP_matrix[:, 0, i]  # absent category is not included
            ar_m = self.AR_matrix[:, 0, i]
            table_iou_list[1].append(self.mr4(ap_m[ap_m > -1]))
            table_iou_list[2].append(self.mr4(ar_m[ar_m > -1]))

        table_area_list = [['Area'] + self.area_name, ['AP'], ['Recall']]
        for i in range(self.A):
            ap_m = self.AP_matrix[:, i, :]
            ar_m = self.AR_matrix[:, i, :]
            table_area_list[1].append(self.mr4(ap_m[ap_m > -1]))
            table_area_list[2].append(self.mr4(ar_m[ar_m > -1]))

        table_c = AsciiTable(table_c_list)
        table_iou = AsciiTable(table_iou_list)
        table_area = AsciiTable(table_area_list)

        if self.all_points:
            print()
            table_mpp = AsciiTable(table_mpp_list)
            print(table_mpp.table)

        print()
        print(table_c.table)  # bug, can not print '\n', or table is not perfect
        print()
        print(table_iou.table)
        print()
        print(table_area.table)


    def draw_curve(self):
        print('\nDrawing precision-recall curves...')
        save_path = f'coco_improved/{self.iou_type}'
        os.makedirs(save_path, exist_ok=True)

        for c in range(self.C):
            print(f'\r{c}/{self.C}, {self.NAMES[c]:>15}', end='')

            mAP = self.mr4(self.AP_matrix[c, 0, :])
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f'{self.NAMES[c]}, mAP={mAP}', size=16, color='red')

            for t in range(self.T):
                recall = np.cumsum(self.r_record[c][0][t]).tolist()
                recall.insert(0, 0.)  # insert 0. to supplement the base point
                r_last = recall[-1]
                precision = self.p_record[c][0][t].tolist()
                precision.insert(0, 1.)

                # Every time we plot, we should use plt APIs to reset all things, or it will reuse
                # the last plot window, and may cause bugs.
                plt.subplot(3, 4, t + 1)
                plt.title(f'iou threshold: {self.iou_thre[t]}', size=12, color='black')
                plt.xlim(0, r_last)
                plt.xlabel('Recall', size=12)
                plt.ylim(0, 1.1)
                plt.ylabel('Precision', size=12)
                plt.tick_params(labelsize=12)  # set tick font size

                ap = self.AP_matrix[c, 0, t]
                p_max, r_max, s_max, mpp, ff = self.MPP_matrix[c][0][t].tolist()

                # draw the MPP rectangle
                plt.hlines(p_max, xmin=0, xmax=r_max, color='blue', linestyles='dashed')
                plt.vlines(r_max, ymin=0, ymax=p_max, color='blue', linestyles='dashed')
                plt.text(r_last, 1.05, f'AP={ap:.3f}', ha='right', va='top', fontsize=12, color='black')
                plt.text(r_max * 0.1, max(p_max - 0.1, 0.2), f'MPP={mpp:.3f}\nFF={ff:.3f}',
                         ha='left', va='top', fontsize=12, color='blue')

                # draw the max recall point
                plt.text(r_last * 1.05, -0.1, f'{r_last:.2f}', ha='center', va='bottom',
                         fontsize=12, color='black', rotation=15)

                # draw the score < 0.05 area
                # hatch: ('/', '//', '-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
                shadow = plt.bar(x=r_last / 2, height=precision[-1], width=r_last,
                                 hatch='//', color='white', edgecolor='grey')

                # draw the s_max point
                plt.scatter(r_max, p_max, color='red')
                plt.text(r_max, p_max, f'{s_max:.2f}', ha='left', va='bottom', fontsize=12, color='red')

                plt.plot(recall, precision, color='black')

            # loc: ('upper right', 'lower left', 'center', 'lower center', (0.4, 0.5) ...)
            fig.legend(handles=[shadow], labels=['Area where detects are filtered.'], loc='upper right', fontsize=12)

            plt.tight_layout()  # resolve the overlapping issue when using subplot()
            plt.savefig(f'{save_path}/{c + 1}_{self.NAMES[c]}.jpg')
            plt.close()

        print()
    
    def draw_pr_curve_for_iou(self, iou_threshold):
        # 找到用户指定的IoU阈值在self.iou_thre中的索引
        if iou_threshold not in self.iou_thre:
            raise ValueError(f'IoU threshold {iou_threshold} is not in the predefined thresholds.')

        t_index = self.iou_thre.index(iou_threshold)

        print(f'\nDrawing precision-recall and F1 curves for IoU={iou_threshold}...')
        save_path = f'coco_improved/{self.iou_type}'
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(15, 10))

        # 绘制PR曲线
        plt.subplot(2, 1, 1)  # PR曲线放在上面
        plt.title(f'Precision-Recall Curves for IoU={iou_threshold}', size=16, color='red')
        plt.xlabel('Recall', size=12)
        plt.ylabel('Precision', size=12)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.1)

        # 初始化F1分数存储
        f1_scores = []

        for c in range(self.C):
            print(f'\rProcessing category: {self.NAMES[c]}', end='')
            mAP = self.AP_matrix[c, 0, t_index]

            recall = np.cumsum(self.r_record[c][0][t_index]).tolist()
            recall.insert(0, 0.)  # insert 0. to supplement the base point
            precision = self.p_record[c][0][t_index].tolist()
            precision.insert(0, 1.)

            # 如果recall最后一个值小于1.0，则插入 [recall[-1], 1.0] 区间的点，步长为0.05
            if recall[-1] < 1.0:
                next_value = recall[-1] + 0.01
                while next_value < 1.0:
                    recall.append(next_value)
                    precision.append(0.0)
                    next_value += 0.01
                
                recall.append(1.0)
                precision.append(0.0)
            plt.plot(recall, precision, label=f'{self.NAMES[c]} (AP={mAP:.3f})')

            # 计算F1分数
            f1 = [(2 * p * r) / (p + r + np.spacing(1)) for p, r in zip(precision, recall)]
            f1_scores.append(f1)

        plt.legend(loc='lower left', fontsize=8)
        plt.savefig(f'{save_path}/pr_curve_iou_{iou_threshold}.jpg')

    def draw_precision_curve_for_iou(self, category_index):
        """
        绘制指定类别在不同IoU阈值下的Precision曲线。
        
        :param category_index: self.NAMES 列表中的类别索引。
        """
        if category_index < 0 or category_index >= len(self.NAMES):
            raise ValueError(f'Invalid category index {category_index}. It should be in range 0 to {len(self.NAMES) - 1}.')

        print(f'\nDrawing precision curves for category {self.NAMES[category_index]}...')
        save_path = f'coco_improved/{self.iou_type}'
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.title(f'Precision Curve for {self.NAMES[category_index]}', size=16, color='red')
        plt.xlabel('IoU Threshold', size=12)
        plt.ylabel('Precision', size=12)
        plt.xlim(min(self.iou_thre), max(self.iou_thre))
        plt.ylim(0, 1.1)

        # 提取所有IoU阈值下的precision
        precisions = [self.AP_matrix[category_index, 0, t] for t in range(self.T)]

        plt.plot(self.iou_thre, precisions, marker='o', label=f'{self.NAMES[category_index]}')

        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{save_path}/precision_curve_{self.NAMES[category_index]}.jpg')
        plt.close()

        print(f'Precision curve for {self.NAMES[category_index]} has been saved.')

    def draw_precision_curve_for_confidence(self, category_index, iou_threshold):
        """
        绘制指定类别在不同Confidence阈值下的Precision曲线。

        :param category_index: self.NAMES 列表中的类别索引。
        :param iou_threshold: 使用的IoU阈值。
        """
        if category_index < 0 or category_index >= len(self.NAMES):
            raise ValueError(f'Invalid category index {category_index}. It should be in range 0 to {len(self.NAMES) - 1}.')

        if iou_threshold not in self.iou_thre:
            raise ValueError(f'IoU threshold {iou_threshold} is not in the predefined thresholds.')

        t_index = self.iou_thre.index(iou_threshold)
        print(f'\nDrawing precision curve for category {self.NAMES[category_index]} at IoU={iou_threshold}...')

        save_path = f'coco_improved/{self.iou_type}'
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.title(f'Precision vs Confidence Curve for {self.NAMES[category_index]} (IoU={iou_threshold})', size=16, color='red')
        plt.xlabel('Confidence Threshold', size=12)
        plt.ylabel('Precision', size=12)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.1)

        # 提取该类别在给定IoU阈值下的所有检测结果的分数和匹配信息
        scores = np.concatenate([aa['dt_score'] for aa in self.match_record[category_index][0]])
        dt_matched = np.concatenate([aa['dt_match'][t_index] for aa in self.match_record[category_index][0]])
        dt_ignore = np.concatenate([aa['dt_ignore'][t_index] for aa in self.match_record[category_index][0]])

        sorted_indices = np.argsort(-scores)
        scores_sorted = scores[sorted_indices]
        tp_sorted = np.logical_and(dt_matched, np.logical_not(dt_ignore))[sorted_indices]
        fp_sorted = np.logical_and(np.logical_not(dt_matched), np.logical_not(dt_ignore))[sorted_indices]

        tp_cumsum = np.cumsum(tp_sorted).astype(dtype='float32')
        fp_cumsum = np.cumsum(fp_sorted).astype(dtype='float32')
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.spacing(1))

        # 绘制Precision vs Confidence曲线
        plt.plot(scores_sorted, precisions, label=f'{self.NAMES[category_index]}')

        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{save_path}/precision_vs_confidence_{self.NAMES[category_index]}_iou_{iou_threshold}.jpg')
        plt.close()

        print(f'Precision vs Confidence curve for {self.NAMES[category_index]} at IoU={iou_threshold} has been saved.')

    def draw_recall_curve_for_confidence(self, category_index, iou_threshold):
        """
        绘制指定类别在不同Confidence阈值下的Recall曲线。

        :param category_index: self.NAMES 列表中的类别索引。
        :param iou_threshold: 使用的IoU阈值。
        """
        if category_index < 0 or category_index >= len(self.NAMES):
            raise ValueError(f'Invalid category index {category_index}. It should be in range 0 to {len(self.NAMES) - 1}.')

        if iou_threshold not in self.iou_thre:
            raise ValueError(f'IoU threshold {iou_threshold} is not in the predefined thresholds.')

        t_index = self.iou_thre.index(iou_threshold)
        print(f'\nDrawing recall curve for category {self.NAMES[category_index]} at IoU={iou_threshold}...')

        save_path = f'coco_improved/{self.iou_type}'
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.title(f'Recall vs Confidence Curve for {self.NAMES[category_index]} (IoU={iou_threshold})', size=16, color='blue')
        plt.xlabel('Confidence Threshold', size=12)
        plt.ylabel('Recall', size=12)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)

        # 提取该类别在给定IoU阈值下的所有检测结果的分数和匹配信息
        scores = np.concatenate([aa['dt_score'] for aa in self.match_record[category_index][0]])
        dt_matched = np.concatenate([aa['dt_match'][t_index] for aa in self.match_record[category_index][0]])
        dt_ignore = np.concatenate([aa['dt_ignore'][t_index] for aa in self.match_record[category_index][0]])

        sorted_indices = np.argsort(-scores)
        scores_sorted = scores[sorted_indices]
        tp_sorted = np.logical_and(dt_matched, np.logical_not(dt_ignore))[sorted_indices]

        num_gt = sum([aa['num_gt'] for aa in self.match_record[category_index][0]])
        recall_cumsum = np.cumsum(tp_sorted).astype(dtype='float32') / num_gt

        # 绘制Recall vs Confidence曲线
        plt.plot(scores_sorted, recall_cumsum, label=f'{self.NAMES[category_index]}')

        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{save_path}/recall_vs_confidence_{self.NAMES[category_index]}_iou_{iou_threshold}.jpg')
        plt.close()

        print(f'Recall vs Confidence curve for {self.NAMES[category_index]} at IoU={iou_threshold} has been saved.')

    def draw_f1_curve_for_confidence(self, category_index, iou_threshold):
        """
        绘制指定类别在不同Confidence阈值下的F1 Score曲线。

        :param category_index: self.NAMES 列表中的类别索引。
        :param iou_threshold: 使用的IoU阈值。
        """
        if category_index < 0 or category_index >= len(self.NAMES):
            raise ValueError(f'Invalid category index {category_index}. It should be in range 0 to {len(self.NAMES) - 1}.')

        if iou_threshold not in self.iou_thre:
            raise ValueError(f'IoU threshold {iou_threshold} is not in the predefined thresholds.')

        t_index = self.iou_thre.index(iou_threshold)
        print(f'\nDrawing F1 score curve for category {self.NAMES[category_index]} at IoU={iou_threshold}...')

        save_path = f'coco_improved/{self.iou_type}'
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.title(f'F1 Score vs Confidence Curve for {self.NAMES[category_index]} (IoU={iou_threshold})', size=16, color='green')
        plt.xlabel('Confidence Threshold', size=12)
        plt.ylabel('F1 Score', size=12)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)

        # 提取该类别在给定IoU阈值下的所有检测结果的分数和匹配信息
        scores = np.concatenate([aa['dt_score'] for aa in self.match_record[category_index][0]])
        dt_matched = np.concatenate([aa['dt_match'][t_index] for aa in self.match_record[category_index][0]])
        dt_ignore = np.concatenate([aa['dt_ignore'][t_index] for aa in self.match_record[category_index][0]])

        sorted_indices = np.argsort(-scores)
        scores_sorted = scores[sorted_indices]
        tp_sorted = np.logical_and(dt_matched, np.logical_not(dt_ignore))[sorted_indices]
        fp_sorted = np.logical_and(np.logical_not(dt_matched), np.logical_not(dt_ignore))[sorted_indices]

        tp_cumsum = np.cumsum(tp_sorted).astype(dtype='float32')
        fp_cumsum = np.cumsum(fp_sorted).astype(dtype='float32')
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.spacing(1))

        num_gt = sum([aa['num_gt'] for aa in self.match_record[category_index][0]])
        recalls = tp_cumsum / num_gt

        f1_scores = (2 * precisions * recalls) / (precisions + recalls + np.spacing(1))

        # 绘制F1 Score vs Confidence曲线
        plt.plot(scores_sorted, f1_scores, label=f'{self.NAMES[category_index]}')

        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{save_path}/f1_vs_confidence_{self.NAMES[category_index]}_iou_{iou_threshold}.jpg')
        plt.close()

        print(f'F1 Score vs Confidence curve for {self.NAMES[category_index]} at IoU={iou_threshold} has been saved.')

    def get_curves_data_for_iou(self, iou_threshold):
        """
        获取所有类别在给定IoU阈值下的PR曲线, P曲线, R曲线, F1曲线的x, y值，并返回数据字典格式.
        同时将结果导出为JSON格式的文件.

        :param iou_threshold: 用于生成曲线的IoU阈值.
        :return: 包含所有类别曲线数据的字典.
        """
        if iou_threshold not in self.iou_thre:
            raise ValueError(f'IoU threshold {iou_threshold} is not in the predefined thresholds.')

        t_index = self.iou_thre.index(iou_threshold)
        print(f'\nGenerating curves data for IoU={iou_threshold}...')

        f1_scores, precision, recall, PR_recall, PR_precision = [], [], [], [], []
        precision_mean, recall_mean, f1_mean, PR_precision_mean = [], [], [], []

        for c in range(self.C):
            # 提取Recall和Precision
            rec = np.cumsum(self.r_record[c][0][t_index]).tolist()
            rec.insert(0, 0.)  # 插入起点

            prec = self.p_record[c][0][t_index].tolist()
            prec.insert(0, 1.)

            # 如果 recall 的最后一个值小于 1.0，插入额外的点
            if rec[-1] < 1.0:
                next_value = rec[-1] + 0.01
                while next_value < 1.0:
                    rec.append(next_value)
                    prec.append(0.0)
                    next_value += 0.01

                rec.append(1.0)
                prec.append(0.0)

            # 计算F1分数
            f1 = [(2 * p * r) / (p + r + np.spacing(1)) for p, r in zip(prec, rec)]

            # 保存PR曲线数据
            PR_recall.append(rec)
            PR_precision.append(prec)

            # 保存Precision曲线数据 (基于Confidence)
            scores = np.concatenate([aa['dt_score'] for aa in self.match_record[c][0]])
            dt_matched = np.concatenate([aa['dt_match'][t_index] for aa in self.match_record[c][0]])
            dt_ignore = np.concatenate([aa['dt_ignore'][t_index] for aa in self.match_record[c][0]])

            sorted_indices = np.argsort(-scores)
            scores_sorted = scores[sorted_indices]
            tp_sorted = np.logical_and(dt_matched, np.logical_not(dt_ignore))[sorted_indices]
            fp_sorted = np.logical_and(np.logical_not(dt_matched), np.logical_not(dt_ignore))[sorted_indices]

            tp_cumsum = np.cumsum(tp_sorted).astype(dtype='float32')
            fp_cumsum = np.cumsum(fp_sorted).astype(dtype='float32')
            precisions_conf = tp_cumsum / (tp_cumsum + fp_cumsum + np.spacing(1))

            num_gt = sum([aa['num_gt'] for aa in self.match_record[c][0]])
            recalls_conf = tp_cumsum / num_gt

            f1_scores_conf = (2 * precisions_conf * recalls_conf) / (precisions_conf + recalls_conf + np.spacing(1))

            # 将 float32 类型转换为普通的 Python float 类型
            precision.append([float(p) for p in precisions_conf])
            recall.append([float(r) for r in recalls_conf])
            f1_scores.append([float(f) for f in f1_scores_conf])

        # 计算平均值，按照垂直维度 (axis=1)
        precision_mean = [float(np.mean(p, axis=0)) for p in np.array(precision).T]
        recall_mean = [float(np.mean(r, axis=0)) for r in np.array(recall).T]
        f1_mean = [float(np.mean(f, axis=0)) for f in np.array(f1_scores).T]
        # PR_precision_mean = [float(np.mean(p, axis=0)) for p in np.array(PR_precision).T]
        PR_precision_mean = [float(np.mean(p, axis=0)) for p in np.array(PR_precision).T]

        from scipy.interpolate import interp1d
        # Step 1: 生成新的PR_recall
        new_recall = np.arange(0, 1.01, 0.01)  # 从0到1，步长为0.01的数组

        # Step 2: 插值处理PR_precision，使其与new_recall长度一致
        new_precision = []

        for prec, rec in zip(PR_precision, PR_recall):
            # 创建插值函数，确保在原先recall位置的precision值保持不变
            interp_func = interp1d(rec, prec, kind='linear', fill_value='extrapolate')
            # 使用新的recall值进行插值
            interpolated_prec = interp_func(new_recall)
            # 添加到新的precision列表中
            new_precision.append(interpolated_prec.tolist())

        PR_precision_mean = [float(np.mean(p, axis=0)) for p in np.array(new_precision).T]

        # 从 AP_matrix 和 AR_matrix 获取给定 IoU 阈值的 AP 和 Recall 数据
        ap_values = self.AP_matrix[:, 0, t_index]
        recall_values = self.AR_matrix[:, 0, t_index]

        result_dict = {
            'precision': float(ap_values.tolist()[0]),  # 给定IoU阈值的AP数据
            'recall': float(recall_values.tolist()[0])  # 给定IoU阈值的Recall数据
        }

        # 组织数据字典
        data_to_save = {
            "names": [{"id": key, "name": value} for key, value in enumerate(self.NAMES)],
            "confidence": scores_sorted.tolist(),
            "f1": {
                "f1_scores": f1_scores,  # F1-score 数据
                "f1_mean": f1_mean  # F1-score 平均值
            },
            "precision": {
                "precision": precision,  # Precision 数据
                "precision_mean": precision_mean  # Precision 平均值
            },
            "recall": {
                "recall": recall,  # Recall 数据
                "recall_mean": recall_mean  # Recall 平均值
            },
            "pr": {
                'recall': new_recall.tolist(),
                'precision': new_precision,
                'precision_mean': PR_precision_mean  # 可能需要重新计算 mean
            },
            'result_dict': result_dict,  # 只包含给定IoU阈值的数据
        }

        return replace_nan_with_zero(data_to_save)