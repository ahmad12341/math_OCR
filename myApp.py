from PIL import Image, ImageOps
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import torch
import numpy as np
import time
from typing import Tuple, Union, List, Dict, Any
from utils import *
import cv2
from cnstd import CnStd
from copy import copy, deepcopy
import cv2
import re
from itertools import chain
import tqdm
import time


det_modelv2 = CnOcr()

images_fps = [
    # '/home/dev/ocr_ea/data set/1.jpg',
    # '/home/dev/ocr_ea/data set/2.PNG',
    # '/home/dev/ocr_ea/data set/3.PNG',
    # '/home/dev/ocr_ea/data set/4.PNG',
    # '/home/dev/ocr_ea/data set/5.PNG',
    # '/home/dev/ocr_ea/data set/6.PNG',
    # '/home/dev/ocr_ea/data set/7.PNG',
    # '/home/dev/ocr_ea/data set/8.png',
    # '/home/dev/ocr_ea/data set/9.png',
    # '/home/dev/ocr_ea/data set/10.png',
    # '/home/dev/ocr_ea/data set/11.png',
    # '/home/dev/ocr_ea/data set/12.png',
    # '/home/dev/ocr_ea/data set/13.png',
    # '/home/dev/ocr_ea/data set/14.png',
    # '/home/dev/ocr_ea/data set/15.png',
    # '/home/dev/ocr_ea/data set/16.png',
    # '/home/dev/ocr_ea/data set/17.png',
    # '/home/dev/ocr_ea/data set/18.png',  
    # '/home/dev/ocr_ea/data set/19.png',
    # '/home/dev/ocr_ea/data set/20.png',
    # '/home/dev/ocr_ea/data set/21.png',
    # '/home/dev/ocr_ea/data set/22.png',
    # '/home/dev/ocr_ea/data set/23.png',
    # '/home/dev/ocr_ea/data set/24.png',
    # '/home/dev/ocr_ea/data set/25.png',
    # '/home/dev/ocr_ea/data set/26.png',
    # '/home/dev/ocr_ea/data set/27.png',    
    # '/home/dev/ocr_ea/data set/28.png',
    # '/home/dev/ocr_ea/data set/29.png',
    # '/home/dev/ocr_ea/data set/30.png',
    # '/home/dev/ocr_ea/data set/31.png',
    # '/home/dev/ocr_ea/data set/32.png',
    # '/home/dev/ocr_ea/data set/33.png',
    # '/home/dev/ocr_ea/data set/34.png',
    # '/home/dev/ocr_ea/data set/35.png',
    # '/home/dev/ocr_ea/data set/36.png', 
    # '/home/dev/ocr_ea/data set/37.png',
    # '/home/dev/ocr_ea/data set/38.png',
    # '/home/dev/ocr_ea/data set/39.png',
    # '/home/dev/ocr_ea/data set/40.png',
    # '/home/dev/ocr_ea/data set/41.png',
    # '/home/dev/ocr_ea/data set/42.png',
    # '/home/dev/ocr_ea/data set/43.png',
    # '/home/dev/ocr_ea/data set/44.png',
    # '/home/dev/ocr_ea/data set/45.png',   
    # '/home/dev/ocr_ea/data set/46.png',
    # '/home/dev/ocr_ea/data set/47.png',
    # '/home/dev/ocr_ea/data set/48.png',
    # '/home/dev/ocr_ea/data set/49.png',
    # '/home/dev/ocr_ea/data set/50.png',   
    # '/home/dev/ocr_ea/data set/51.png',
    # '/home/dev/ocr_ea/data set/52.png',
    # '/home/dev/ocr_ea/data set/53.png',
    # '/home/dev/ocr_ea/data set/54.png',
    # '/home/dev/ocr_ea/data set/55.png',   
    # '/home/dev/ocr_ea/data set/56.png',
    # '/home/dev/ocr_ea/data set/57.png',
    # '/home/dev/ocr_ea/data set/58.png',
    # '/home/dev/ocr_ea/data set/59.png',
    # '/home/dev/ocr_ea/data set/60.png',   
    '/home/dev/ocr_ea/data set/61.png',
    '/home/dev/ocr_ea/data set/62.png',
    '/home/dev/ocr_ea/data set/63.png',
    '/home/dev/ocr_ea/data set/64.png',
    '/home/dev/ocr_ea/data set/65.png',   
    '/home/dev/ocr_ea/data set/66.png',
    '/home/dev/ocr_ea/data set/67.png',
    '/home/dev/ocr_ea/data set/68.png',
    '/home/dev/ocr_ea/data set/69.png',
    '/home/dev/ocr_ea/data set/70.png',   
    '/home/dev/ocr_ea/data set/71.png',
    '/home/dev/ocr_ea/data set/72.png',
    '/home/dev/ocr_ea/data set/73.png',
    '/home/dev/ocr_ea/data set/74.png',
    '/home/dev/ocr_ea/data set/75.png',
    '/home/dev/ocr_ea/data set/76.png',
    '/home/dev/ocr_ea/data set/77.png',
    '/home/dev/ocr_ea/data set/78.png',         
]

# img_fp = "/home/dev/ocr_ea/testingRecognition/1.jpg"

model_path = "/home/dev/.pix2text/1.1/mfd-onnx/mfd-v20240618.onnx"

det_model_name = 'ch_PP-OCRv3_det'
det_model_backend = 'onnx'
context = 'cpu'
det_model_fp = None
det_root = '/home/dev/.cnstd'

languages=('en', 'ch_sim')

max_width_expand_ratio = 0.3
embed_ratio_threshold = 0.6
embed_sep = (' $', '$ ')
isolated_sep = ('$$\n', '\n$$')
line_sep = '\n'
auto_line_break = True

predictor = DetectionPredictor()
predictor.setup_model(model_path)

processor = TrOCRProcessor.from_pretrained('breezedeus/pix2text-mfr')
model = ORTModelForVision2Seq.from_pretrained('breezedeus/pix2text-mfr', use_cache=False)


det_model = CnStd(det_model_name, model_backend=det_model_backend, context=context, model_fp=det_model_fp, root=det_root,)


def _split_line_image(line_box, embed_mfs):
    line_box = line_box[0]
    if not embed_mfs:
        return [{'position': line_box.int().tolist(), 'type': 'text'}]
    embed_mfs.sort(key=lambda x: x['position'][0])

    outs = []
    start = int(line_box[0])
    xmax, ymin, ymax = int(line_box[2]), int(line_box[1]), int(line_box[-1])
    for mf in embed_mfs:
        _xmax = min(xmax, int(mf['position'][0]) + 1)
        if start + 8 < _xmax:
            outs.append({'position': [start, ymin, _xmax, ymax], 'type': 'text'})
        start = int(mf['position'][2])
        if _xmax >= xmax:
            break
    if start < xmax:
        outs.append({'position': [start, ymin, xmax, ymax], 'type': 'text'})
    return outs

def recognize_only(img: np.ndarray):
    try:
        return det_modelv2.ocr_for_single_line(img)
    except Exception as error:
        print("An exception occurred:", error)
        return {'text': '', 'score': 0.0}

def _post_process(outs1):
    match_pairs = [
        (',', ',，'),
        ('.', '.。'),
        ('?', '?？'),
    ]
    formula_tag = '^[（\(]\d+(\.\d+)*[）\)]$'

    def _match(a1, a2):
        matched = False
        for b1, b2 in match_pairs:
            if a1 in b1 and a2 in b2:
                matched = True
                break
        return matched
    
    for idx, line_boxes in enumerate(outs1):
        if (
            any([_lang in ('ch_sim', 'ch_tra') for _lang in languages])
            and len(line_boxes) > 1
            and line_boxes[-1]['type'] == 'text'
            and line_boxes[-2]['type'] != 'text'
        ):
            if line_boxes[-1]['text'].lower() == 'o':
                line_boxes[-1]['text'] = '。'
        if len(line_boxes) > 1:
            # 去掉边界上多余的标点
            for _idx2, box in enumerate(line_boxes[1:]):
                if (
                    box['type'] == 'text'
                    and line_boxes[_idx2]['type'] == 'embedding'
                ):  # if the current box is text and the previous box is embedding
                    if _match(line_boxes[_idx2]['text'][-1], box['text'][0]) and (
                        not line_boxes[_idx2]['text'][:-1].endswith('\\')
                        and not line_boxes[_idx2]['text'][:-1].endswith(r'\end')
                    ):
                        line_boxes[_idx2]['text'] = line_boxes[_idx2]['text'][:-1]
            # 把 公式 tag 合并到公式里面去
            for _idx2, box in enumerate(line_boxes[1:]):
                if (
                    box['type'] == 'text'
                    and line_boxes[_idx2]['type'] == 'isolated'
                ):  # if the current box is text and the previous box is embedding
                    if y_overlap(line_boxes[_idx2], box, key='position') > 0.9:
                        print("i m here", str(box['text']))
                        if re.match(formula_tag, str(box['text'])):
                            print("coming here and there")
                            # 去掉开头和结尾的括号
                            tag_text = box['text'][1:-1]
                            line_boxes[_idx2]['text'] = line_boxes[_idx2][
                                'text'
                            ] + ' \\tag{{{}}}'.format(tag_text)
                            new_xmax = max(
                                line_boxes[_idx2]['position'][2][0],
                                box['position'][2][0],
                            )
                            line_boxes[_idx2]['position'][1][0] = line_boxes[_idx2][
                                'position'
                            ][2][0] = new_xmax
                            box['text'] = ''
        outs1[idx] = [box for box in line_boxes if str(box.get('text')).strip()]
    return outs1

# def _cal_scores(self, outs):
#     if isinstance(outs, GenerateBeamEncoderDecoderOutput):
#         mean_probs = outs.sequences_scores.exp().tolist()
#     elif isinstance(outs, GenerateEncoderDecoderOutput):
#         logits = torch.stack(outs.scores, dim=1)
#         scores = torch.softmax(logits, dim=-1).max(dim=2).values

#         mean_probs = []
#         for idx, example in enumerate(scores):
#             cur_length = int(
#                 (outs.sequences[idx] != self.processor.tokenizer.pad_token_id).sum()
#             )
#             assert cur_length > 1
#             # 获得几何平均值。注意：example中的第一个元素对应sequence中的第二个元素
#             mean_probs.append(
#                 float((example[: cur_length - 1] + 1e-8).log().mean().exp())
#             )
#     else:
#         raise Exception(f'unprocessed output type: {type(outs)}')

#     return mean_probs


def _one_batch(img_list, rec_config):
    rec_config = rec_config or {}
    pixel_values = processor(images=img_list, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)

    # mean_probs = _cal_scores(generated_ids)
    
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True) # add inference score later

    assert len(img_list) == len(generated_text)

    final_out = []
    for text in zip(generated_text):
        final_out.append({'text': text})
    return final_out

r = open("mytext", 'w')

def runInference():
    elapsed_time = 0
    for img_fp in images_fps: 
        start_time = 0
        end_time = 0
        start_time = time.time()
        rec_config = None
        batch_size = 1
        resized_shape  = 768
        box_margin = 0
        dedup_thrsh = 0.1
        crop_patches = []
        analyzer_outs = []
        outs = [] # gets the all four corners of the bounding box of mfd detection
        mf_outs = []
        results = []

        img = read_img(img_fp, return_type='Image')
        

        w, h = img.size
        ratio = resized_shape / w
        resized_shape = (int(h * ratio), resized_shape)

    ## mathematical formula detection
        batch_results = predictor(img, True) # mfd-v20240618.onnx

    ## getting the bounding box (all four corners)
        for res in batch_results:
            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            # print("boxes", boxes)
            scores = res.boxes.conf.cpu().numpy().tolist()
            labels = res.boxes.cls.cpu().int().numpy().tolist()
            categories = res.names
            height, width = res.orig_shape
            one_out = []
            for box, score, label in zip(boxes, scores, labels):
                box = expand_box_by_margin(box, box_margin, (height, width))
                box = xyxy24p(box, ret_type=np.array)
                one_out.append({'box': box, 'score': score, 'type': categories[label]})

            one_out = sortboxes(one_out, key='box')
            one_out = dedup_boxes(one_out, threshold=dedup_thrsh)
            outs.append(one_out)

        if len(outs) == 1:
            # print("yolo outs", outs[0])
            analyzer_outs = outs[0]
        else:
            analyzer_outs = outs

    ## crop the image 
        for mf_box_info in analyzer_outs:
            box = mf_box_info['box']
            xmin, ymin, xmax, ymax = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )    
            crop_patch = img.crop((xmin, ymin, xmax, ymax))
            crop_patches.append(crop_patch)
            # crop_patch.show()

        input_imgs = prepare_imgs(crop_patches)
        # print("input_imgs", input_imgs)

    # mathematical formula recoginition
        for i in tqdm.tqdm(range(0, len(input_imgs), batch_size)):
            part_imgs = input_imgs[i : i + batch_size]
            results.extend(_one_batch(part_imgs, rec_config))
        # print("mf_results are", results)
        assert len(results) == len(analyzer_outs)

        
        for mf_box_info, patch_out in zip(analyzer_outs, results):
            text = patch_out['text']
            mf_outs.append(
                {
                    'type': mf_box_info['type'],
                    'text': text,
                    'position': mf_box_info['box'],
                    # 'score': patch_out['score'],
                }
            )

    # masks the image based on the detected bounding box
        masked_img = np.array(img.copy())
        for mf_box_info in analyzer_outs:
            if mf_box_info['type'] in ('isolated', 'embedding'):
                box = mf_box_info['box']
                xmin, ymin = max(0, int(box[0][0]) - 1), max(0, int(box[0][1]) - 1)
                xmax, ymax = (
                    min(img.size[0], int(box[2][0]) + 1),
                    min(img.size[1], int(box[2][1]) + 1),
                )
                masked_img[ymin:ymax, xmin:xmax, :] = 255
        masked_img = Image.fromarray(masked_img)
        # masked_img.show()

    # detects the text in the image
        detOut = det_model.detect(np.array(img)) #, resized_shape=resized_shape

        for out in detOut['detected_texts']:
            out['position'] = out.pop('box')
        # print("detOut",detOut) 

        box_infos = []
        for line_box_info in detOut['detected_texts']:
            # crop_img_info['box'] 可能是一个带角度的矩形框，需要转换成水平的矩形框
            _text_box = rotated_box_to_horizontal(line_box_info['position'])
            if not is_valid_box(_text_box, min_height=8, min_width=2):
                continue
            box_infos.append({'position': _text_box})          
        box_infos: list[dict] = adjust_line_width(
                text_box_infos=box_infos,
                formula_box_infos=mf_outs,
                img_width=img.size[0],
                max_expand_ratio=max_width_expand_ratio,
        )
        # print("box_infos", box_infos)    
        box_infos = remove_overlap_text_bbox(box_infos, mf_outs)
        # print("box_infos after", box_infos)

        def _to_iou_box(ori):
            return torch.tensor([ori[0][0], ori[0][1], ori[2][0], ori[2][1]]).unsqueeze(
                0
            )
            
        total_text_boxes = []
        for line_box_info in box_infos:
            _line_box = _to_iou_box(line_box_info['position'])
            _embed_mfs = []
            for mf_box_info in mf_outs:
                if mf_box_info['type'] == 'embedding':
                    _mf_box = _to_iou_box(mf_box_info['position'])
                    overlap_area_ratio = float(
                        box_partial_overlap(_line_box, _mf_box).squeeze()
                    )
                    if overlap_area_ratio >= embed_ratio_threshold or (
                        overlap_area_ratio > 0
                        and y_overlap(line_box_info, mf_box_info, key='position')
                        > embed_ratio_threshold
                    ):
                        _embed_mfs.append(
                            {
                                'position': _mf_box[0].int().tolist(),
                                'text': mf_box_info['text'],
                                'type': mf_box_info['type'],
                            }
                        )

            ocr_boxes = _split_line_image(_line_box, _embed_mfs)
            total_text_boxes.extend(ocr_boxes)      

        outsText = copy(mf_outs)
        # print("outs copy", outsText)
        for box in total_text_boxes:
            box['position'] = list2box(*box['position'])
            outsText.append(box)
        outsText = sort_boxes(outsText, key='position')
        # print("outsText after sort", outsText)
        outsText = [merge_adjacent_bboxes(bboxes) for bboxes in outsText]
        # print("outsText after merge", outsText)
        max_height_expand_ratio = 0.2

        outsText = adjust_line_height(
            outsText, img.size[1], max_expand_ratio=max_height_expand_ratio
        )
        # print("outsText", outsText)
        for line_idx, line_boxes in enumerate(outsText):
            for box in line_boxes:
                if box['type'] != 'text':
                    continue
                bbox = box['position']
                xmin, ymin, xmax, ymax = (
                    int(bbox[0][0]),
                    int(bbox[0][1]),
                    int(bbox[2][0]),
                    int(bbox[2][1]),
                )
                crop_patch = np.array(masked_img.crop((xmin, ymin, xmax, ymax)))
                # print("crop_patch", crop_patch)
                part_res = recognize_only(crop_patch)
                # print("part_res", part_res)
                box['text'] = part_res['text']
                box['score'] = part_res['score']
            # print("type of box", type(box))
            outsText[line_idx] = [box for box in line_boxes if str(box['text']).strip()]

        # print("outsText after", outsText)
        outsText = _post_process(outsText)
        outsText = list(chain(*outsText))
        # print("outsText after chains", outsText)

        outsText = merge_line_texts(
        outsText,
        auto_line_break,
        line_sep,
        embed_sep,
        isolated_sep,
        )
        end_time = time.time()

        elapsed_time = elapsed_time + ( end_time - start_time)

        mylife = outsText.replace("\\\\", "\\")
        print("********************")
        print(mylife)
        r.write("********************")
        r.write(mylife)
        r.write("********************")
        print("********************")
        print(elapsed_time)
    

def main():
    # for img_fp in images_fps:
    runInference()

if __name__ == '__main__':
    main()


