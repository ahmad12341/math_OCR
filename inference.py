from utils import *
import numpy as np
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq

mfr_model_path = '/home/dev/ocr_ea/testingRecognition/pix2text-mfr'

processor = TrOCRProcessor.from_pretrained(mfr_model_path)
model = ORTModelForVision2Seq.from_pretrained(mfr_model_path, use_cache=False)

def extractBoundingBoxes(batch_results):
    box_margin = 0
    dedup_thrsh = 0.1
    outs = []
    for res in batch_results: ## getting the bounding box (all four corners)
        boxes = res.boxes.xyxy.cpu().numpy().tolist()
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
        one_out = dedup_boxes(one_out, threshold=dedup_thrsh) # removes the duplicates / overlapping boxes
        outs.append(one_out)

    if len(outs) == 1:
        return outs[0]
    else:
        return outs
    
def cropImages(img, analyzer_outs):
    crop_patches = []
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
    return crop_patches


def mask_the_image(img, analyzer_outs):
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
    return masked_img

def _one_batch(img_list, rec_config):
    rec_config = rec_config or {}
    pixel_values = processor(images=img_list, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)
    
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True) # add inference score later

    assert len(img_list) == len(generated_text)

    final_out = []
    for text in zip(generated_text):
        final_out.append({'text': text})
    return final_out

def recognize_formula(input_imgs, analyzer_outs):
    batch_size = 1
    rec_config = None
    mf_outs = []
    results = []
    for i in range(0, len(input_imgs), batch_size):
        part_imgs = input_imgs[i : i + batch_size]
        results.extend(_one_batch(part_imgs, rec_config))

    assert len(results) == len(analyzer_outs)
 
    for mf_box_info, patch_out in zip(analyzer_outs, results):
        text = patch_out['text']
        mf_outs.append({'type': mf_box_info['type'],'text': text,'position': mf_box_info['box'],})
    return mf_outs