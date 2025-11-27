from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import torch

# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


def get_temporal_features_torch(features):
    """
    Computes temporal features from given features (only temporal tokens, consistent with training).
    
    Parameters:
    features (torch.Tensor): Input features to process. Shape: [t, s, c] where t is time frames, s is spatial patches, c is feature dim.
    
    Returns:
    torch.Tensor: Temporal features. Shape: [100, c] - only temporal tokens, padded to 100.
    """
    t, s, c = features.shape
    
    # Compute temporal tokens as the mean along the spatial axis (same as training)
    temporal_tokens = torch.mean(features, dim=1)  # [t, c] - 对空间维度平均
    
    # Padding to 100
    padding_size = 100 - t
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    elif t > 100:
        # If more than 100 frames, truncate
        temporal_tokens = temporal_tokens[:100]
    
    return temporal_tokens.half()  # [100, c]


def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.
    (Kept for backward compatibility)

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens


def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens. (Deprecated: now fixed to 100, kept for backward compatibility)

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Prepare question string for the model
    # Use fixed 100 tokens (consistent with training, temporal only)
    num_tokens = 100
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * num_tokens + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * num_tokens

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Preprocess video frames and get image tensor
    image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

    # Move image tensor to GPU and reduce precision to half
    image_tensor = image_tensor.half().cuda()

    # Generate video temporal features (only temporal tokens, consistent with training)
    with torch.no_grad():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
    video_spatio_temporal_features = get_temporal_features_torch(frame_features)  # [100, 1024]

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs