import sys,os
import pandas as pd 
import argparse

os.environ['TRANSFORMERS_CACHE']='../hub/'
os.environ['HF_DATASETS_CACHE']='../datasets/'
import evaluate
import torch 
import torch.nn.functional as F
import whisper
from whisper.model import AudioEncoder
from transformers import WhisperProcessor
from models.cogenav import CoGenAV
from utils import get_audio_video

def cogenav_forward(self, x):
    # By simply modifying the inference code of the Whisper encoder, we integrated CoGenAV into Whisper.
    is_cogenav = x.shape[1]!=80 and  x.shape[1]!=128
    if not is_cogenav:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

    x = x.permute(0, 2, 1)
    assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
    x = (x + self.positional_embedding).to(x.dtype)

    # Check if is_cogenav input and self has the adapter attribute and if it is a callable object.
    if is_cogenav and hasattr(self, 'adapter') and callable(self.adapter)  :
        x = self.adapter(x, x)

    for block in self.blocks:
        x = block(x)
    x = self.ln_post(x)
    return x

AudioEncoder.forward = cogenav_forward

def clc_wer(pred_strs,label_strs):
    # It's best to execute the following in China: export HF_ENDPOINT=https://hf-mirror.com
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_strs, references=label_strs)
    return wer 

def main(args):
    df = pd.read_csv(args.asr_data_csv)
    data = df.to_dict(orient='records')

    noise_wav = args.noise_wav
    root_dir = args.root_dir

    cogenav_ckpt = args.cogenav_ckpt

    beam_size = args.beam_size
    model_size = args.model_size
    with_noise = not args.without_noise
    input_type = args.input_type

    # 1. load cogenav model
    cfg_file = f'config/{model_size}.yaml'
    if cogenav_ckpt is None:
        cogenav_ckpt = f'weights/{model_size}_cogen.pt'
    cogenav = CoGenAV(cfg_file=cfg_file, model_tensor=cogenav_ckpt).cuda()
    cogenav.eval()
    print("------cogenav model arch------")
    print(cogenav)

    # 2. load whisper model
    SR_Head = 'medium' if model_size=="large" else "small"
    SR_Head = whisper.load_model(SR_Head, download_root="weights/whisper/")

    # 3. Add an adapter to the SR_Head encoder, and the inference function has been modified.
    SR_Head.encoder.adapter = cogenav.adapter.half()
    SR_Head.eval()

    print("------whisper model arch------")
    print(SR_Head)

    #4. inference options such as beam_size and processor
    options = whisper.DecodingOptions(beam_size=beam_size, language='en', task='transcribe', fp16=True, without_timestamps=True, patience=2.0)
    image_mean = 0.421 if model_size=="large" else 0.0
    image_std = 0.165 if model_size=="large" else 1.0

    #5. infer & eval
    pred_strs, label_strs = [], []
    for data_info in data:
        vidname = data_info["asr_video"]
        gt_text = data_info["asr_text"]
        audio, video, length, _ = get_audio_video(root_dir, (vidname, noise_wav), None, 
                            image_mean=image_mean, image_std=image_std, with_noise=with_noise)
        with torch.no_grad():
            if input_type == "whisper_a":
                input_ids = audio.cuda()
            elif input_type == "cogenav_av":
                input_ids = cogenav(video.cuda(), audio.cuda(),).permute(0, 2, 1)
            elif input_type == "cogenav_v":
                input_ids = cogenav(video.cuda(), None).permute(0, 2, 1)
            elif input_type == "cogenav_a":
                input_ids = cogenav(None, audio.cuda()).permute(0, 2, 1)

        result = whisper.decode(SR_Head, input_ids, options)[0]

        pred = result.text
        if input_type == "whisper_a":
            pred = pred.lower().replace(".","").replace("?","").replace("!","").replace(",","")+"."# whisper_a zeroshot,去除大小写，标点符号等影响
        label_strs.append(gt_text)
        pred_strs.append(pred) 

        print(vidname)
        print(f"ref :{gt_text}")
        print(f"pred:{pred}")

    wer = clc_wer(pred_strs, label_strs)
    print(wer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Model Testing")
    parser.add_argument("--asr_data_csv", type=str,default="data/asr_data_test.csv", help="Path to the ASR data CSV file")
    parser.add_argument("--noise_wav", type=str,default='data/noise/babble/babble_all_2nd.wav', help="Path to the noise WAV file")
    parser.add_argument("--root_dir", type=str, default='/mnt/workspace/detao.bdt/datasets/avsr/', help="Root directory for data")
    
    # Add model and config parameters
    parser.add_argument("--cogenav_ckpt", type=str, default=None, help="Path to the cogenav model tensor")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for decoding")
    parser.add_argument("--model_size", type=str,default='large', choices=['base', 'large'], help="Model name (base or large)")
    parser.add_argument("--without_noise", action='store_true', help="Do not use noise in input signal")
    parser.add_argument("--input_type", type=str, default='cogenav_av',choices=['cogenav_av', 'cogenav_v', 'cogenav_a','whisper_a'], help="Input type for model")

    args = parser.parse_args()

    main(args)
