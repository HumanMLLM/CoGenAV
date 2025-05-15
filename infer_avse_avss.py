import sys,os
import pandas as pd 
import argparse
import numpy as np
os.environ['TRANSFORMERS_CACHE']='../hub/'
os.environ['HF_DATASETS_CACHE']='../datasets/'
import evaluate
import torch 
import torch.nn.functional as F
import whisper
from whisper.model import AudioEncoder
from transformers import WhisperProcessor
from models.cogenav import CoGenAV
from utils import get_audio_video,process_test_file
from  models.sepformer import build_Sepformer
import torch
from scipy.io import wavfile

def load_model_parameters(checkpoint_path):
    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    # 提取模型的状态字典
    model_state_dict = checkpoint['state_dict']  # 或者使用 checkpoint['model']，具体取决于保存时的结构

    # 打印参数的名称和形状
    selected_tensors={}
    for name, param in model_state_dict.items():
        if "feature_extract."  in name :continue
        print(f"Parameter Name: {name}, Shape: {param.shape}")
        new_key = name.replace("model.","")
        selected_tensors[new_key] = param
    return selected_tensors
def main(args):

    noise_wav = args.noise_wav
    root_dir = args.root_dir
    save_dir = args.save_dir

    cogenav_ckpt = args.cogenav_ckpt
    sepformer_ckpt= args.sepformer_ckpt

    model_size = args.model_size
    task_type = args.task_type

    # 1. load cogenav model
    assert model_size=="base"
    cfg_file = f'config/{model_size}.yaml'
    cogenav = CoGenAV(cfg_file=cfg_file, model_tensor=cogenav_ckpt).cuda()
    cogenav.eval()
    print("------cogenav model arch------")
    print(cogenav)

    # 2. load sepformer model
    sepformer_head = build_Sepformer().cuda()
    sepformer_head.load_state_dict(torch.load(sepformer_ckpt),strict=True)
    sepformer_head.eval()

    #4. prepare data for avss and avse
    if task_type=="avss":
        data = process_test_file()
    else:
        df = pd.read_csv(args.asr_data_csv)
        data = df.to_dict(orient='records')

    #5. infer
    for data_info in data:
        if task_type=="avss":
            vidname = data_info[0]
            merge_wav = os.path.join(root_dir,data_info[1].replace(".mp4",".wav"))
            snr = data_info[-1]
        else:
            vidname = data_info["asr_video"]
            snr = 0
            merge_wav= noise_wav
        # if not ("6343252661930009508" in  vidname and  "00092" in  vidname):continue
        audio, video, length, audio_mix = get_audio_video(root_dir, (vidname, merge_wav), None, 
                            image_mean=0.0, image_std=1.0, with_noise=True,SNR=snr)
        video = video[:,:,:,:,:length].cuda()
        audio_mix = audio_mix[:length*640]
        #this is only for infer demo
        audio_mix = torch.tensor(audio_mix).unsqueeze(0).cuda()
        print(vidname,snr,merge_wav,video.shape,length,audio_mix.shape)
        with torch.no_grad():
            lip_feature = cogenav(video, None,use_upsampler=False)
            sep_wav = sepformer_head.forward(audio_mix, lip_feature)

        mix_name = os.path.join(save_dir,task_type,vidname.replace(".mp4",f"_mix_snr{snr}.wav"))
        pred_name = os.path.join(save_dir,task_type,vidname.replace(".mp4",f"_pred_snr{snr}_.wav"))
        os.makedirs(os.path.dirname(pred_name),exist_ok=True)

        wavfile.write(pred_name, 16000, (sep_wav.detach().cpu().numpy()[0] * 32768).astype(np.int16))
        wavfile.write(mix_name, 16000, (audio_mix.cpu().numpy()[0] * 32768).astype(np.int16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Model Testing")
    parser.add_argument("--asr_data_csv", type=str,default="data/asr_data_test.csv", help="Path to the ASR data CSV file")
    parser.add_argument("--noise_wav", type=str,default='data/noise/babble/babble_all_2nd.wav', help="Path to the noise WAV file")
    parser.add_argument("--root_dir", type=str, default='/mnt/workspace/detao.bdt/datasets/avsr/', help="Root directory for data")

    parser.add_argument("--save_dir", type=str, default='wav_vis/', help="Root directory for data")

    # Add model and config parameters
    parser.add_argument("--cogenav_ckpt", type=str, default="weights/base_cogen.pt", help="Path to the cogenav model tensor")
    parser.add_argument("--sepformer_ckpt", type=str, default="weights/sepformer_head.pt", help="Path to the cogenav model tensor")
    parser.add_argument("--model_size", type=str,default='base', choices=['base',], help="Model name (base or large)")
    parser.add_argument("--task_type", type=str, default='avss',choices=['avss', 'avse'], help="Input type for model")
    args = parser.parse_args()
    main(args)

