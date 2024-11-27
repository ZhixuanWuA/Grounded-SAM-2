import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import spacy
import gc
import psutil

# def out_for_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / (1024 * 1024)  # 转换为 MB
#     print(f"---------------------------------Out_for_Memory Usage------------------------------------: {mem:.2f} MB")

# def doc_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / (1024 * 1024)  # 转换为 MB
#     print(f"---------------------------------Load_Doc_Memory Usage------------------------------------: {mem:.2f} MB")

# def frame_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / (1024 * 1024)  # 转换为 MB
#     print(f"---------------------------------Load_Frame_Memory Usage----------------------------------: {mem:.2f} MB")
# def mask1_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / (1024 * 1024)  # 转换为 MB
#     print(f"---------------------------------Mask1_Memory Usage---------------------------------------: {mem:.2f} MB")

# def mask2_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / (1024 * 1024)  # 转换为 MB
#     print(f"---------------------------------Mask2_Memory Usage---------------------------------------: {mem:.2f} MB")

# def video_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / (1024 * 1024)  # 转换为 MB
#     print(f"---------------------------------Video_Memory Usage---------------------------------------: {mem:.2f} MB")



device = "cuda:0"
# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

def load_label_exclusions(label_file_path):
    exclusions = []
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            exclusion_path = os.path.join('/home/zhangshaoxing/cv/datasets/Videomme/all_names.txt',line.strip())
            exclusions.append(exclusion_path)
    return exclusions

def process_videos(base_video_dir, output_dir, qa_json_path, sam2_checkpoint, model_cfg, grounding_model_id, json_type):
    # Load the qa.json data
    with open(qa_json_path, "r") as f:
        qa_data = json.load(f)

    # List of words to exclude
    question_words = ["what", "who", "which", "where", "when", "why", "how many", "video"]

    # Use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize models
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    
    exclusions = load_label_exclusions("/home/zhangshaoxing/cv/datasets/Videomme/all_names.txt")
    print(f'exclusions:{exclusions}')

    # List all video folders in the base directory
    video_folders = [
        os.path.join(base_video_dir, d) for d in os.listdir(base_video_dir)
        if os.path.isdir(os.path.join(base_video_dir, d))
    ]
    # out_for_memory_usage()

    # Process each video folder
    for video_dir in video_folders:
        if video_dir in exclusions:
            print(f"Skipping video folder: {video_dir}")
            continue  # 如果文件夹在排除列表中，跳过该文件夹
        # Get the video name
        video_name = os.path.basename(video_dir)

        # Find matching questions in qa.json
        # matching_questions = [
        #     (item["question_id"], item["question"])
        #     for item in qa_data if item["videoID"] == video_name
        # ]
        matching_questions = [
            (item["question_id"], item["question"], item.get("options", []))
            for item in qa_data if item["videoID"] == video_name
        ]

        if matching_questions:
            for question_id, input_question, options in matching_questions:
                print(f"Found question for video '{video_name}': {input_question}")
                
                # Process the input question to extract nouns and pronouns
                doc = nlp(input_question) # memory_usage()
                nouns_and_pronouns = [
                    token.text for token in doc 
                    if token.pos_ in ["NOUN"] and token.text.lower() not in question_words
                ]
                text = '. '.join(nouns_and_pronouns).strip()

                # For json2 files, if no nouns in question, extract from options
                if json_type == "/home/zhangshaoxing/cv/datasets/Videomme/videomme/test-00000-of-nonobject.json" and not text and options:
                    print(f"No nouns found in the question for video '{video_name}', extracting from options.")
                    option_nouns = set()
                    for option in options:
                        doc_option = nlp(option)
                        for token in doc_option:
                            if token.pos_ == "NOUN" and token.text.lower() not in question_words:
                                option_nouns.add(token.text)
                    if option_nouns:
                        text = ', '.join(option_nouns)

                if not text:  # Check if text is empty after processing
                    text += 'people.'
                if text and not text.endswith('.'):
                    text += '.'
                print("Final text for processing:", text)
                # doc_memory_usage()


                # Create output directories using question_id
                question_output_dir = os.path.join(output_dir, question_id)
                CommonUtils.creat_dirs(question_output_dir)

                # Create directories for masks, json data, and results
                mask_data_dir = os.path.join(question_output_dir, "mask_data")
                json_data_dir = os.path.join(question_output_dir, "json_data")
                result_dir = os.path.join(question_output_dir, "result")
                CommonUtils.creat_dirs(mask_data_dir)
                CommonUtils.creat_dirs(json_data_dir)
                CommonUtils.creat_dirs(result_dir)

                # Get the output video path
                output_video_path = os.path.join(question_output_dir, f"{question_id}_output.mp4")
                if os.path.exists(output_video_path):
                    print(f"Output video already exists: {output_video_path}, skipping.")
                    continue
                frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".png"]]
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].replace('frame', '')))
                inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
                # frame_memory_usage()

                step = 20
                sam2_masks = MaskDictionaryModel()
                PROMPT_TYPE_FOR_VIDEO = "mask"
                objects_count = 0

                for start_frame_idx in range(0, len(frame_names), step):
                    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
                    image = Image.open(img_path)
                    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)
                    
                    image_base_name = frame_names[start_frame_idx].split(".")[0]
                    mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")

                    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                    print("Input IDs:", inputs.input_ids)
    
                    with torch.no_grad():
                        outputs = grounding_model(**inputs)
                    

                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=0.25,
                        text_threshold=0.25,
                        target_sizes=[image.size[::-1]]
                        # target_sizes=[noisy_image.size[::-1]]
                    )
                    
                    if not results or not results[0]["boxes"].numel():
                        print(f"Skipping frame {start_frame_idx} due to empty detection results.")
                        continue

                    image_predictor.set_image(np.array(image.convert("RGB")))
                    input_boxes = results[0]["boxes"]
                    OBJECTS = results[0]["labels"]

                    masks, scores, logits = image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )

                    if masks.ndim == 2:
                        masks = masks[None]
                        scores = scores[None]
                        logits = logits[None]
                    elif masks.ndim == 4:
                        masks = masks.squeeze(1)
                    
                   
                    if mask_dict.promote_type == "mask":
                        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
                    else:               
                        raise NotImplementedError("SAM 2 video predictor only supports mask prompts")

                    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
                    video_predictor.reset_state(inference_state)

                    if len(mask_dict.labels) == 0:
                        print("No object detected in the frame, skipping frame {}".format(start_frame_idx))
                        continue

                    for object_id, object_info in mask_dict.labels.items():
                        video_predictor.add_new_mask(
                            inference_state,
                            start_frame_idx,
                            object_id,
                            object_info.mask,
                        )
                    # mask1_memory_usage()

                    video_segments = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                        frame_masks = MaskDictionaryModel()

                        for i, out_obj_id in enumerate(out_obj_ids):
                            out_mask = (out_mask_logits[i] > 0.0)
                            object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id))
                            object_info.update_box()
                            frame_masks.labels[out_obj_id] = object_info
                            image_base_name = frame_names[out_frame_idx].split(".")[0]
                            frame_masks.mask_name = f"mask_{image_base_name}.npy"
                            frame_masks.mask_height = out_mask.shape[-2]
                            frame_masks.mask_width = out_mask.shape[-1]

                        video_segments[out_frame_idx] = frame_masks
                        # sam2_masks = copy.deepcopy(frame_masks)
                        sam2_masks = frame_masks
                    # mask2_memory_usage() 

                    for frame_idx, frame_masks_info in video_segments.items():
                        mask = frame_masks_info.labels
                        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                        for obj_id, obj_info in mask.items():
                            mask_img[obj_info.mask == True] = obj_id
                            # mask_img[obj_info.mask == True] = 0

                        mask_img = mask_img.numpy().astype(np.uint16)
                        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

                        json_data = frame_masks_info.to_dict()
                        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                        with open(json_data_path, "w") as f:
                            json.dump(json_data, f)
                    # video_memory_usage()
                    del image, inputs, outputs, results, masks, scores, logits, video_segments, frame_masks, out_mask ,object_info, mask, mask_img, json_data, mask_dict, frame_masks_info, out_mask_logits
                    torch.cuda.empty_cache()
                    gc.collect()

                CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)
                create_video_from_images(result_dir, output_video_path, frame_rate=10)

                print(f"Output video saved to: {output_video_path}")

                # torch.cuda.empty_cache()  # 清理 GPU 缓存
                # gc.collect()
                del sam2_masks
                torch.cuda.empty_cache()  # 清理 GPU 缓存
                gc.collect()

    del matching_questions, inference_state
    torch.cuda.empty_cache()
    gc.collect()


# Example usage
if __name__ == "__main__":
    # base_video_dir = "/home/hanjiale/cv/Detect/Grounded-SAM-2-main/test/videos"
    # output_dir = "/home/hanjiale/cv/Detect/Grounded-SAM-2-main/test/output"
    base_video_dir = "/home/zhangshaoxing/cv/datasets/Videomme/videomme/frames"
    output_dir = "/home/zhangshaoxing/cv/datasets/Videomme/videomme/sam2_addmaskblack_withoutlabel_result"
    qa_json_path = "/home/zhangshaoxing/cv/datasets/Videomme/videomme/test-00000-of-object.json"
    qa2_json_path = "/home/zhangshaoxing/cv/datasets/Videomme/videomme/test-00000-of-nonobject.json"
    sam2_checkpoint = "/home/zhangshaoxing/cv/model/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    grounding_model_id = "/home/zhangshaoxing/cv/model/grounding-dino-tiny"

    process_videos(base_video_dir, output_dir, qa_json_path, sam2_checkpoint, model_cfg, grounding_model_id, json_type=qa2_json_path)
