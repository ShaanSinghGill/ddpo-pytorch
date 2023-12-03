from PIL import Image
import io
import numpy as np
import torch


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]] * len(image_batch),
                "answers": [[f"The image contains {prompt}"] for prompt in prompt_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += np.array(response_data["precision"]).squeeze().tolist()
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn



def counter_reward():
    import math
    from transformers import pipeline
    def postprocess(results):
        count = [1 for i in range(len(results))]
        for i in range(len(results)):
            for j in range(len(results)):
                if i < j:
                    boxi = results[i]["box"]
                    boxj = results[j]["box"]
                    ixmin = boxi["xmin"]
                    iymin = boxi["ymin"]
                    jxmin = boxj["xmin"]
                    jymin = boxj["ymin"]
                    if math.sqrt(abs(jxmin-ixmin)**2 + abs(jymin-iymin)**2) <= 30:
                        count[j] = 0
        temp = [results[i] for i in range(len(results)) if count[i] == 1]
        goods = []
        bads = []
        for i in range(len(temp)):
            box = temp[i]["box"]
            xdist = box["xmax"] - box["xmin"]
            ydist = box["ymax"] - box["ymin"]
            if max(xdist, ydist) / min(xdist, ydist) > 1.75:
                bads.append(temp[i])
            else:
                goods.append(temp[i])
        return goods, bads
    
    model_name = "zoheb/yolos-small-balloon"
    obj_detector = pipeline("object-detection", model = model_name, threshold = 0.9, device='cuda:1')
    base_reward = 100
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        
        scores = []
        for i in range(len(images)):
            results = obj_detector(images[i])
            goods, bads = postprocess(results)
            total = len(goods) + len(bads)
            if total == 0:
                scores.append(0)
                continue
            reward = base_reward
            ###get the number
            promptv = 5
            if "1" in prompts[i]:
                promptv = 1
            if "5" in prompts[i]: 
                promptv = 5
            if "3" in prompts[i]: 
                promptv = 3
            if "7" in prompts[i]: 
                promptv = 7
            if "9" in prompts[i]: 
                promptv = 9
            #penalize for being off
            reward = reward - 15 * abs(total-promptv)#100 * (abs(total-promptv)/promptv)
            #penalize for having bads
            reward = reward - 5 * len(bads)
            scores.append(reward)
        return scores, {}
        
    return _fn




def counter_rewardV2():
    from transformers import DetrImageProcessor, DetrForObjectDetection
    device = 'cuda:1'
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model.to(device)
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        
        scores = []
        
        for image, prompt in zip(images, prompts):
            expected_count = 5 if "5" in prompt else 1
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()} 
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]], device=device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
            reward = 100
            count = 0
            invalid = 0
            for label, box in zip(results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                label_str = model.config.id2label[label.item()]
                width = box[2] - box[0]
                height = box[3] - box[1]
                aspect_ratio = width / height
                if ("ball" in label_str or "orange" in label_str) and 0.5 <= aspect_ratio <= 2:
                    count += 1
                elif ("ball" in label_str or "orange" in label_str): 
                    invalid += 1
            if count != expected_count: 
                diff = abs(expected_count-count)
                reward -= 20 * diff
                reward -= 5 * invalid
                
            scores.append(reward)
        return scores, {}
        
    return _fn
