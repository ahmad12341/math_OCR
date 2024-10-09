from typing import Tuple, Union, List, Dict, Any
import torch
import numpy as np
from PIL import Image, ImageOps
from copy import deepcopy
from collections import Counter, defaultdict
import re
from cnocr import CnOcr




_deepcopy_dispatch = d = {}

def expand_box_by_margin(xyxy, box_margin, shape_hw):
    xmin, ymin, xmax, ymax = [float(_x) for _x in xyxy]
    xmin = max(0, xmin - box_margin)
    ymin = max(0, ymin - box_margin)
    xmax = min(shape_hw[1], xmax + box_margin)
    ymax = min(shape_hw[0], ymax + box_margin)
    return [xmin, ymin, xmax, ymax]


def xyxy24p(x, ret_type=torch.Tensor):
    xmin, ymin, xmax, ymax = [float(_x) for _x in x]
    out = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    if ret_type is not None:
        return ret_type(out).reshape((4, 2))
    return out


def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K


def _compare_box(box1, box2, key):
    # 从上到下，从左到右
    # box1, box2 to: [xmin, ymin, xmax, ymax]
    box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
    box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]

    def y_iou():
        # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
        # 判断是否有交集
        if box1[3] <= box2[1] or box2[3] <= box1[1]:
            return 0
        # 计算交集的高度
        y_min = max(box1[1], box2[1])
        y_max = min(box1[3], box2[3])
        return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))

    if y_iou() > 0.5:
        return box1[0] - box2[0]
    else:
        return box1[1] - box2[1]
    

def box_partial_overlap(box1, cond_box):
    """ intersection / area(cand_box) """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    # area1 = box_area(box1.T)
    area2 = box_area(cond_box.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], cond_box[:, 2:]) - torch.max(box1[:, None, :2], cond_box[:, :2])).clamp(0).prod(2)
    return inter / (area2[:, None] + 1e-6)  # iou = inter / area2


def dedup_boxes(one_out, threshold):
    def _to_iou_box(ori):
        return torch.tensor([ori[0][0], ori[0][1], ori[2][0], ori[2][1]]).unsqueeze(0)

    keep = [True] * len(one_out)
    for idx, info in enumerate(one_out):
        box = _to_iou_box(info['box'])
        if not keep[idx]:
            continue
        for l in range(idx + 1, len(one_out)):
            if not keep[l]:
                continue
            box2 = _to_iou_box(one_out[l]['box'])
            v1 = float(box_partial_overlap(box, box2).squeeze())
            v2 = float(box_partial_overlap(box2, box).squeeze())
            if v1 >= v2:
                if v1 >= threshold:
                    keep[l] = False
            else:
                if v2 >= threshold:
                    keep[idx] = False
                    break

    return [info for idx, info in enumerate(one_out) if keep[idx]]

def get_same_line_boxes(anchor, total_boxes):
    line_boxes = [anchor]
    for box in total_boxes:
        if box['line_number'] >= 0:
            continue
        if max([y_overlap(box, l_box) for l_box in line_boxes]) > 0.1:
            line_boxes.append(box)
    return line_boxes


def sort_and_filter_line_boxes(line_boxes, key):
    if len(line_boxes) <= 1:
        return line_boxes

    allowed_max_overlay_x = 20

    def find_right_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        right_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][0, 0] >= anchor[key][2, 0] - allowed_max
        ]
        if not right_boxes:
            return None
        right_boxes = sorted(
            right_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=True)
            ),
        )
        return right_boxes[0]

    def find_left_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        left_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][2, 0] <= anchor[key][0, 0] + allowed_max
        ]
        if not left_boxes:
            return None
        left_boxes = sorted(
            left_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=False)
            ),
        )
        return left_boxes[-1]

    res_boxes = [line_boxes[0]]
    anchor = res_boxes[0]
    line_number = anchor['line_number']

    while True:
        right_box = find_right_box(anchor)
        if right_box is None:
            break
        right_box['line_number'] = line_number
        res_boxes.append(right_box)
        anchor = right_box

    anchor = res_boxes[0]
    while True:
        left_box = find_left_box(anchor)
        if left_box is None:
            break
        left_box['line_number'] = line_number
        res_boxes.insert(0, left_box)
        anchor = left_box

    return res_boxes

def sortboxes(
    dt_boxes: List[Union[Dict[str, Any], Tuple[np.ndarray, float]]],
    key: Union[str, int] = 'box',
) -> List[Union[Dict[str, Any], Tuple[np.ndarray, float]]]:
    """
    Sort resulting boxes in order from top to bottom, left to right
    args:
        dt_boxes(array): list of dict or tuple, box with shape [4, 2]
    return:
        sorted boxes(array): list of dict or tuple, box with shape [4, 2]
    """
    _boxes = sorted(dt_boxes, key=cmp_to_key(lambda x, y: _compare_box(x, y, key)))
    return _boxes


def sort_boxes(boxes: List[dict], key='position') -> List[List[dict]]:
    # 按y坐标排序所有的框
    boxes.sort(key=lambda box: box[key][0, 1])
    for box in boxes:
        box['line_number'] = -1  # 所在行号，-1表示未分配

    def get_anchor():
        anchor = None
        for box in boxes:
            if box['line_number'] == -1:
                anchor = box
                break
        return anchor

    lines = []
    while True:
        anchor = get_anchor()
        if anchor is None:
            break
        anchor['line_number'] = len(lines)
        line_boxes = get_same_line_boxes(anchor, boxes)
        line_boxes = sort_and_filter_line_boxes(line_boxes, key)
        lines.append(line_boxes)

    return lines


def read_img(
    path: Union[str], return_type='Tensor'
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """

    Args:
        path (str): image file path
        return_type (str): 返回类型；
            支持 `Tensor`，返回 torch.Tensor；`ndarray`，返回 np.ndarray；`Image`，返回 `Image.Image`

    Returns: RGB Image.Image, or np.ndarray / torch.Tensor, with shape [Channel, Height, Width]
    """
    assert return_type in ('Tensor', 'ndarray', 'Image')
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert('RGB')  # 识别旋转后的图片（pillow不会自动识别）
    if return_type == 'Image':
        return img
    img = np.array(img)
    if return_type == 'ndarray':
        return img
    return torch.tensor(img.transpose((2, 0, 1)))


def prepare_imgs(imgs: List[Union[str, Image.Image]]) -> List[Image.Image]:
    output_imgs = []
    for img in imgs:
        if isinstance(img, (str)):
            img = read_img(img, return_type='Image')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
        else:
            raise ValueError(f'Unsupported image type: {type(img)}')
        output_imgs.append(img)

    return output_imgs

def rotated_box_to_horizontal(box):
    """将旋转框转换为水平矩形。

    :param box: [4, 2]，左上角、右上角、右下角、左下角的坐标
    """
    xmin = min(box[:, 0])
    xmax = max(box[:, 0])
    ymin = min(box[:, 1])
    ymax = max(box[:, 1])
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

def is_valid_box(box, min_height=8, min_width=2) -> bool:
    """判断box是否有效。
    :param box: [4, 2]，左上角、右上角、右下角、左下角的坐标
    :param min_height: 最小高度
    :param min_width: 最小宽度
    :return: bool, 是否有效
    """
    return (
        box[0, 0] + min_width <= box[1, 0]
        and box[1, 1] + min_height <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_width
        and box[3, 1] >= box[0, 1] + min_height
    )


def y_overlap(box1, box2, key='position'):
    # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
    if key:
        box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
        box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    else:
        box1 = [box1[0][0], box1[0][1], box1[2][0], box1[2][1]]
        box2 = [box2[0][0], box2[0][1], box2[2][0], box2[2][1]]
    # 判断是否有交集
    if box1[3] <= box2[1] or box2[3] <= box1[1]:
        return 0
    # 计算交集的高度
    y_min = max(box1[1], box2[1])
    y_max = min(box1[3], box2[3])
    return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))


def x_overlap(box1, box2, key='position'):
    # 计算它们在x轴上的IOU: Interaction / min(width1, width2)
    if key:
        box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
        box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    else:
        box1 = [box1[0][0], box1[0][1], box1[2][0], box1[2][1]]
        box2 = [box2[0][0], box2[0][1], box2[2][0], box2[2][1]]
    # 判断是否有交集
    if box1[2] <= box2[0] or box2[2] <= box1[0]:
        return 0
    # 计算交集的宽度
    x_min = max(box1[0], box2[0])
    x_max = min(box1[2], box2[2])
    return (x_max - x_min) / max(1, min(box1[2] - box1[0], box2[2] - box2[0]))


def overlap(box1, box2, key='position'):
    return x_overlap(box1, box2, key) * y_overlap(box1, box2, key)

def adjust_line_width(
    text_box_infos, formula_box_infos, img_width, max_expand_ratio=0.2
):
    """
    如果不与其他 box 重叠，就把 text box 往左右稍微扩展一些（检测出来的 text box 在边界上可能会切掉边界字符的一部分）。
    Args:
        text_box_infos (List[dict]): 文本框信息，其中 'box' 字段包含四个角点的坐标。
        formula_box_infos (List[dict]): 公式框信息，其中 'position' 字段包含四个角点的坐标。
        img_width (int): 原始图像的宽度。
        max_expand_ratio (float): 相对于 box 高度来说的左右最大扩展比率。

    Returns: 扩展后的 text_box_infos。
    """

    def _expand_left_right(box):
        expanded_box = box.copy()
        xmin, xmax = box[0, 0], box[2, 0]
        box_height = box[2, 1] - box[0, 1]
        expand_size = int(max_expand_ratio * box_height)
        expanded_box[3, 0] = expanded_box[0, 0] = max(xmin - expand_size, 0)
        expanded_box[2, 0] = expanded_box[1, 0] = min(xmax + expand_size, img_width - 1)
        return expanded_box

    def _is_adjacent(anchor_box, text_box):
        if overlap(anchor_box, text_box, key=None) < 1e-6:
            return False
        anchor_xmin, anchor_xmax = anchor_box[0, 0], anchor_box[2, 0]
        text_xmin, text_xmax = text_box[0, 0], text_box[2, 0]
        if (
            text_xmin < anchor_xmin < text_xmax < anchor_xmax
            or anchor_xmin < text_xmin < anchor_xmax < text_xmax
        ):
            return True
        return False

    for idx, text_box in enumerate(text_box_infos):
        expanded_box = _expand_left_right(text_box['position'])
        overlapped = False
        cand_boxes = [
            _text_box['position']
            for _idx, _text_box in enumerate(text_box_infos)
            if _idx != idx
        ]
        cand_boxes.extend(
            [_formula_box['position'] for _formula_box in formula_box_infos]
        )
        for cand_box in cand_boxes:
            if _is_adjacent(expanded_box, cand_box):
                overlapped = True
                break
        if not overlapped:
            text_box_infos[idx]['position'] = expanded_box

    return text_box_infos

def list2box(xmin, ymin, xmax, ymax, dtype=float):
    return np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=dtype
    )


def crop_box(text_box, formula_box, min_crop_width=2) -> List[np.ndarray]:
    """
    将 text_box 与 formula_box 相交的部分裁剪掉
    Args:
        text_box ():
        formula_box ():
        min_crop_width (int): 裁剪后新的 text box 被保留的最小宽度，低于此宽度的 text box 会被删除。

    Returns:

    """
    text_xmin, text_xmax = text_box[0, 0], text_box[2, 0]
    text_ymin, text_ymax = text_box[0, 1], text_box[2, 1]
    formula_xmin, formula_xmax = formula_box[0, 0], formula_box[2, 0]

    cropped_boxes = []
    if text_xmin < formula_xmin:
        new_text_xmax = min(text_xmax, formula_xmin)
        if new_text_xmax - text_xmin >= min_crop_width:
            cropped_boxes.append((text_xmin, text_ymin, new_text_xmax, text_ymax))

    if text_xmax > formula_xmax:
        new_text_xmin = max(text_xmin, formula_xmax)
        if text_xmax - new_text_xmin >= min_crop_width:
            cropped_boxes.append((new_text_xmin, text_ymin, text_xmax, text_ymax))

    return [list2box(*box, dtype=None) for box in cropped_boxes]


def remove_overlap_text_bbox(text_box_infos, formula_box_infos):
    """
    如果一个 text box 与 formula_box 相交，则裁剪 text box。
    Args:
        text_box_infos ():
        formula_box_infos ():

    Returns:

    """

    new_text_box_infos = []
    for idx, text_box in enumerate(text_box_infos):
        max_overlap_val = 0
        max_overlap_fbox = None

        for formula_box in formula_box_infos:
            cur_val = overlap(text_box['position'], formula_box['position'], key=None)
            if cur_val > max_overlap_val:
                max_overlap_val = cur_val
                max_overlap_fbox = formula_box

        if max_overlap_val < 0.1:  # overlap 太少的情况不做任何处理
            new_text_box_infos.append(text_box)
            continue
        # if max_overlap_val > 0.8:  # overlap 太多的情况，直接扔掉 text box
        #     continue

        cropped_text_boxes = crop_box(
            text_box['position'], max_overlap_fbox['position']
        )
        if cropped_text_boxes:
            for _box in cropped_text_boxes:
                new_box = deepcopy(text_box)
                new_box['position'] = _box
                new_text_box_infos.append(new_box)

    return new_text_box_infos


def merge_adjacent_bboxes(line_bboxes):
    """
    合并同一行中相邻且足够接近的边界框（bboxes）。
    如果两个边界框在水平方向上的距离小于行的高度，则将它们合并为一个边界框。

    :param line_bboxes: 包含边界框信息的列表，每个边界框包含行号、位置（四个角点的坐标）和类型。
    :return: 合并后的边界框列表。
    """
    merged_bboxes = []
    current_bbox = None

    for bbox in line_bboxes:
        # 如果是当前行的第一个边界框，或者与上一个边界框不在同一行
        if current_bbox is None:
            current_bbox = bbox
            continue

        line_number = bbox['line_number']
        position = bbox['position']
        bbox_type = bbox['type']

        # 计算边界框的高度和宽度
        height = position[2, 1] - position[0, 1]

        # 检查当前边界框与上一个边界框的距离
        distance = position[0, 0] - current_bbox['position'][1, 0]
        if (
            current_bbox['type'] == 'text'
            and bbox_type == 'text'
            and distance <= height
        ):
            # 合并边界框：ymin 取两个框对应值的较小值，ymax 取两个框对应值的较大
            # [text]_[text] -> [text_text]
            ymin = min(position[0, 1], current_bbox['position'][0, 1])
            ymax = max(position[2, 1], current_bbox['position'][2, 1])
            xmin = current_bbox['position'][0, 0]
            xmax = position[2, 0]
            current_bbox['position'] = list2box(xmin, ymin, xmax, ymax)
        else:
            if (
                current_bbox['type'] == 'text'
                and bbox_type != 'text'
                and 0 < distance <= height
            ):
                # [text]_[embedding] -> [text_][embedding]
                current_bbox['position'][1, 0] = position[0, 0]
                current_bbox['position'][2, 0] = position[0, 0]
            elif (
                current_bbox['type'] != 'text'
                and bbox_type == 'text'
                and 0 < distance <= height
            ):
                # [embedding]_[text] -> [embedding][_text]
                position[0, 0] = current_bbox['position'][1, 0]
                position[3, 0] = current_bbox['position'][1, 0]
            # 添加当前边界框，并开始新的合并
            merged_bboxes.append(current_bbox)
            current_bbox = bbox

    if current_bbox is not None:
        merged_bboxes.append(current_bbox)

    return merged_bboxes


def adjust_line_height(bboxes, img_height, max_expand_ratio=0.2):
    """
    基于临近行与行之间间隙，把 box 的高度略微调高（检测出来的 box 可以挨着文字很近）。
    Args:
        bboxes (List[List[dict]]): 包含边界框信息的列表，每个边界框包含行号、位置（四个角点的坐标）和类型。
        img_height (int): 原始图像的高度。
        max_expand_ratio (float): 相对于 box 高度来说的上下最大扩展比率

    Returns:

    """

    def get_max_text_ymax(line_bboxes):
        return max([bbox['position'][2, 1] for bbox in line_bboxes])

    def get_min_text_ymin(line_bboxes):
        return min([bbox['position'][0, 1] for bbox in line_bboxes])

    if len(bboxes) < 1:
        return bboxes

    for line_idx, line_bboxes in enumerate(bboxes):
        next_line_ymin = (
            get_min_text_ymin(bboxes[line_idx + 1])
            if line_idx < len(bboxes) - 1
            else img_height
        )
        above_line_ymax = get_max_text_ymax(bboxes[line_idx - 1]) if line_idx > 0 else 0
        for box in line_bboxes:
            if box['type'] != 'text':
                continue
            box_height = box['position'][2, 1] - box['position'][0, 1]
            if box['position'][0, 1] > above_line_ymax:
                expand_size = min(
                    (box['position'][0, 1] - above_line_ymax) // 3,
                    int(max_expand_ratio * box_height),
                )
                box['position'][0, 1] -= expand_size
                box['position'][1, 1] -= expand_size
            if box['position'][2, 1] < next_line_ymin:
                expand_size = min(
                    (next_line_ymin - box['position'][2, 1]) // 3,
                    int(max_expand_ratio * box_height),
                )
                box['position'][2, 1] += expand_size
                box['position'][3, 1] += expand_size
    return bboxes


def is_chinese(ch):
    """
    判断一个字符是否为中文字符
    """
    return '\u4e00' <= ch <= '\u9fff'

def find_first_punctuation_position(text):
    # 匹配常见标点符号的正则表达式
    pattern = re.compile(r'[,.!?;:()\[\]{}\'\"\\/-]')
    match = pattern.search(text)
    if match:
        return match.start()
    else:
        return len(text)    

def smart_join(str_list, spellchecker=None):
    """
    对字符串列表进行拼接，如果相邻的两个字符串都是中文或包含空白符号，则不加空格；其他情况则加空格
    """

    def contain_whitespace(s):
        if re.search(r'\s', s):
            return True
        else:
            return False

    str_list = [s for s in str_list if s]
    if not str_list:
        return ''
    res = str_list[0]
    for i in range(1, len(str_list)):
        if (is_chinese(res[-1]) and is_chinese(str_list[i][0])) or contain_whitespace(
            res[-1] + str_list[i][0]
        ):
            res += str_list[i]
        elif spellchecker is not None and res.endswith('-'):
            fields = res.rsplit(' ', maxsplit=1)
            if len(fields) > 1:
                new_res, prev_word = fields[0], fields[1]
            else:
                new_res, prev_word = '', res

            fields = str_list[i].split(' ', maxsplit=1)
            if len(fields) > 1:
                next_word, new_next = fields[0], fields[1]
            else:
                next_word, new_next = str_list[i], ''

            punct_idx = find_first_punctuation_position(next_word)
            next_word = next_word[:punct_idx]
            new_next = str_list[i][len(next_word) :]
            new_word = prev_word[:-1] + next_word
            if (
                next_word
                and spellchecker.unknown([prev_word + next_word])
                and spellchecker.known([new_word])
            ):
                res = new_res + ' ' + new_word + new_next
            else:
                new_word = prev_word + next_word
                res = new_res + ' ' + new_word + new_next
        else:
            res += ' ' + str_list[i]
    return res


def cal_block_xmin_xmax(lines, indentation_thrsh):
    total_min_x, total_max_x = min(lines[:, 0]), max(lines[:, 1])
    if lines.shape[0] < 2:
        return total_min_x, total_max_x

    min_x, max_x = min(lines[1:, 0]), max(lines[1:, 1])
    first_line_is_full = total_max_x > max_x - indentation_thrsh
    if first_line_is_full:
        return min_x, total_max_x

    return total_min_x, total_max_x


def merge_line_texts(
    outs: List[Dict[str, Any]],
    auto_line_break: bool = True,
    line_sep='\n',
    embed_sep=(' $', '$ '),
    isolated_sep=('$$\n', '\n$$'),
    spellchecker=None,
) -> str:
    """
    把 Pix2Text.recognize_by_mfd() 的返回结果，合并成单个字符串
    Args:
        outs (List[Dict[str, Any]]):
        auto_line_break: 基于box位置自动判断是否该换行
        line_sep: 行与行之间的分隔符
        embed_sep (tuple): Prefix and suffix for embedding latex; default value is `(' $', '$ ')`
        isolated_sep (tuple): Prefix and suffix for isolated latex; default value is `('$$\n', '\n$$')`
        spellchecker: Spell Checker

    Returns: 合并后的字符串

    """
    if not outs:
        return ''
    out_texts = []
    line_margin_list = []  # 每行的最左边和最右边的x坐标
    isolated_included = []  # 每行是否包含了 `isolated` 类型的数学公式
    line_height_dict = defaultdict(list)  # 每行中每个块对应的高度
    line_ymin_ymax_list = []  # 每行的最上边和最下边的y坐标
    for _out in outs:
        line_number = _out.get('line_number', 0)
        while len(out_texts) <= line_number:
            out_texts.append([])
            line_margin_list.append([100000, 0])
            isolated_included.append(False)
            line_ymin_ymax_list.append([100000, 0])
        cur_text = str(_out['text'])
        cur_type = _out.get('type', 'text')
        box = _out['position']
        if cur_type in ('embedding', 'isolated'):
            sep = isolated_sep if _out['type'] == 'isolated' else embed_sep
            cur_text = sep[0] + cur_text + sep[1]
        if cur_type == 'isolated':
            isolated_included[line_number] = True
            cur_text = line_sep + cur_text + line_sep
        out_texts[line_number].append(cur_text)
        line_margin_list[line_number][1] = max(
            line_margin_list[line_number][1], float(box[2, 0])
        )
        line_margin_list[line_number][0] = min(
            line_margin_list[line_number][0], float(box[0, 0])
        )
        if cur_type == 'text':
            line_height_dict[line_number].append(box[2, 1] - box[1, 1])
            line_ymin_ymax_list[line_number][0] = min(
                line_ymin_ymax_list[line_number][0], float(box[0, 1])
            )
            line_ymin_ymax_list[line_number][1] = max(
                line_ymin_ymax_list[line_number][1], float(box[2, 1])
            )

    line_text_list = [smart_join(o) for o in out_texts]

    for _line_number in line_height_dict.keys():
        if line_height_dict[_line_number]:
            line_height_dict[_line_number] = np.mean(line_height_dict[_line_number])
    _line_heights = list(line_height_dict.values())
    mean_height = np.mean(_line_heights) if _line_heights else None

    default_res = re.sub(rf'{line_sep}+', line_sep, line_sep.join(line_text_list))
    if not auto_line_break:
        return default_res

    line_lengths = [rx - lx for lx, rx in line_margin_list]
    line_length_thrsh = max(line_lengths) * 0.3
    if line_length_thrsh < 1:
        return default_res

    lines = np.array(
        [
            margin
            for idx, margin in enumerate(line_margin_list)
            if isolated_included[idx] or line_lengths[idx] >= line_length_thrsh
        ]
    )
    if lines.shape[0] < 1:
        return default_res
    min_x, max_x = min(lines[:, 0]), max(lines[:, 1])

    indentation_thrsh = (max_x - min_x) * 0.1
    if mean_height is not None:
        indentation_thrsh = 1.5 * mean_height

    min_x, max_x = cal_block_xmin_xmax(lines, indentation_thrsh)

    res_line_texts = [''] * len(line_text_list)
    line_text_list = [(idx, txt) for idx, txt in enumerate(line_text_list) if txt]
    for idx, (line_number, txt) in enumerate(line_text_list):
        if isolated_included[line_number]:
            res_line_texts[line_number] = line_sep + txt + line_sep
            continue

        tmp = txt
        if line_margin_list[line_number][0] > min_x + indentation_thrsh:
            tmp = line_sep + txt
        if line_margin_list[line_number][1] < max_x - indentation_thrsh:
            tmp = tmp + line_sep
        if idx < len(line_text_list) - 1:
            cur_height = line_ymin_ymax_list[line_number][1] - line_ymin_ymax_list[line_number][0]
            next_line_number = line_text_list[idx + 1][0]
            if (
                cur_height > 0
                and line_ymin_ymax_list[next_line_number][0] < line_ymin_ymax_list[next_line_number][1]
                and line_ymin_ymax_list[next_line_number][0] - line_ymin_ymax_list[line_number][1]
                > cur_height
            ):  # 当前行与下一行的间距超过了一行的行高，则认为它们之间应该是不同的段落
                tmp = tmp + line_sep
        res_line_texts[idx] = tmp

    outs = smart_join([c for c in res_line_texts if c], spellchecker)
    return re.sub(rf'{line_sep}+', line_sep, outs)  # 把多个 '\n' 替换为 '\n'
