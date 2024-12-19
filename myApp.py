from flask import Flask, jsonify, request, g
from flask_cors import CORS
# from PIL import Image, ImageOps
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
import os
from LLM import *
from inference import *
from benchmark import *

API_KEY = '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'
UPLOAD_FOLDER = 'uploads'

mfd_model_path = "/home/dev/.pix2text/1.1/mfd-onnx/mfd-v20240618.onnx"
# mfr_model_path = '/home/dev/ocr_ea/testingRecognition/pix2text-mfr'

det_modelv2 = CnOcr()
text_det_model = CnStd('ch_PP-OCRv3_det', model_backend='onnx', context='cpu', model_fp=None, root='/home/dev/.cnstd')
predictor = DetectionPredictor()
predictor.setup_model(mfd_model_path)
# processor = TrOCRProcessor.from_pretrained(mfr_model_path)
# model = ORTModelForVision2Seq.from_pretrained(mfr_model_path, use_cache=False)

# languages=('en', 'ch_sim')
# max_width_expand_ratio = 0.3
# embed_ratio_threshold = 0.6
# embed_sep = (' $', '$ ')
# isolated_sep = ('$$\n', '\n$$')
# line_sep = '\n'
# auto_line_break = True
directory_path = "/home/dev/ocr_ea/benchmarking dataset"
# dict = {}
# def prepare_master(filepath):
#     i = 1
#     with open(filepath, 'w') as file:
#         for root, dirs, files in os.walk(directory_path):
#             for image in files:
#                 dict[image] = json.dumps(i)
#                 i+=1
#         json.dump(dict, file, indent=4)

# myApp = Flask(__name__)
# CORS(myApp)

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
    languages=('en', 'ch_sim')
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
                        if re.match(formula_tag, str(box['text'])):
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

# def _one_batch(img_list, rec_config):
#     rec_config = rec_config or {}
#     pixel_values = processor(images=img_list, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)
    
#     generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True) # add inference score later

#     assert len(img_list) == len(generated_text)

#     final_out = []
#     for text in zip(generated_text):
#         final_out.append({'text': text})
#     return final_out


def runInference(img_fp:str):
    max_width_expand_ratio = 0.3
    embed_ratio_threshold = 0.6
    embed_sep = (' $', '$ ')
    isolated_sep = ('$$\n', '\n$$')
    line_sep = '\n'
    auto_line_break = True
    resized_shape  = 768
    croped_patches = []
    analyzer_outs = []# gets the all four corners of the bounding box of mfd detection
    mf_outs = []
    box_infos = []
    total_text_boxes = []

    img = read_img(img_fp, return_type='Image') # opens an image, adjusts its orientation and converts to RGB
    w, h = img.size
    ratio = resized_shape / w
    resized_shape = (int(h * ratio), resized_shape)
    # resizedImg = img.resize(resized_shape)
    # resizedImg.show()

    batch_results = predictor(img, True) # mfd-v20240618.onnx ## mathematical formula detection
    analyzer_outs = extractBoundingBoxes(batch_results) ## getting the bounding box (all four corners)
    croped_patches = cropImages(img, analyzer_outs) ## crop the image
    input_imgs = prepare_imgs(croped_patches)
    mf_outs = recognize_formula(input_imgs, analyzer_outs) # mathematical formula recoginition
    masked_img = mask_the_image(img, analyzer_outs) # masks the image based on the detected bounding box
    detOut = text_det_model.detect(np.array(masked_img)) #, resized_shape=resized_shape # detects the text in the image

    for out in detOut['detected_texts']:
        out['position'] = out.pop('box')      

    
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
    box_infos = remove_overlap_text_bbox(box_infos, mf_outs)

    def _to_iou_box(ori):
        return torch.tensor([ori[0][0], ori[0][1], ori[2][0], ori[2][1]]).unsqueeze(0)
        
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
    for box in total_text_boxes:
        box['position'] = list2box(*box['position'])
        outsText.append(box)
    outsText = sort_boxes(outsText, key='position')
    outsText = [merge_adjacent_bboxes(bboxes) for bboxes in outsText]
    max_height_expand_ratio = 0.2

    outsText = adjust_line_height(
        outsText, img.size[1], max_expand_ratio=max_height_expand_ratio
    )
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
            crop_patch = masked_img.crop((xmin, ymin, xmax, ymax))
            crop_patch = np.array(crop_patch)
            part_res = recognize_only(crop_patch)
            box['text'] = part_res['text']
            box['score'] = part_res['score']
        outsText[line_idx] = [box for box in line_boxes if str(box['text']).strip()]

    outsText = _post_process(outsText)
    outsText = list(chain(*outsText))

    outsText = merge_line_texts(
    outsText,
    auto_line_break,
    line_sep,
    embed_sep,
    isolated_sep,
    )

    mylife = outsText.replace("'", " ")
    # print("*********ocr out***********")
    # print(repr(mylife))
    # print("*********ocr out***********")
    return mylife



# @myApp.route("/getImage", methods = ['GET', 'POST'])
# def getImage():
#     imageFile = request.files['image']
#     imageName = imageFile.filename
#     fileSavePath = os.path.join(UPLOAD_FOLDER, imageName)
#     imageFile.save(fileSavePath)
#     yourLife = runInference(fileSavePath)
#     category = getEquationType(LLM.llama3_70b, API_KEY,yourLife)
#     print("************category***********")
#     print(category)
#     print("************category***********")
#     response = queryLLM(LLM.llama3_70b, API_KEY, yourLife, category)
#     os.remove(fileSavePath)
#     return (response)



def main():
    # for img_fp in images_fps:
    dict = {}
    # with open("test.json", "w") as file:
    #     for root, dirs, files in os.walk(directory_path):
    #         for image in files:
    #             imagePath = os.path.join(root, image)
    #             yourLife = runInference(imagePath)
    #             dict[image] = json.dumps(yourLife)
    #     #         i+=1
    #     json.dump(dict, file, indent=4)

    master_data = load_json('master.json')
    pipeline_data = load_json('test.json')        
    results = benchmark_pipeline(master_data, pipeline_data)
    print("Benchmarking Results:", results)
        # yourLife = runInference()
        # queryLLM(LLM.llama3_70b, API_KEY, yourLife)



if __name__ == '__main__':
    main()

# prepare_master("master.json")


# if __name__ == "__main__":
#     myApp.run(debug=True, port=9393)