import os
import numpy as np
from decord import VideoReader, cpu
import librosa
from scipy.io import wavfile
import torch
from transformers import WhisperProcessor

try:
    from petrel_client.client import Client
    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False
def get_video_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    def _loader(video_path):
        if _client is not None and 's3:' in video_path:
            video_path = io.BytesIO(_client.get(video_path))
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return vr
    return _loader

video_loader = get_video_loader()
def load_audio(root_dir, wav_path, processor):
    wav_path = wav_path[0]
    wav_path = root_dir + wav_path.replace(".mp4", ".wav")
    audio_array, sampling_rate = librosa.load(wav_path, sr=16000)
    mel_spectrogram = processor.feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]    
    return mel_spectrogram,audio_array

def select_noise(noise_wavs):
    rand_indexes = np.random.randint(0, len(noise_wavs), size=1)
    noise_wav = []
    for x in rand_indexes:
        noise_wav.append(wavfile.read(noise_wavs[x])[1].astype(np.float32))
    return noise_wav[0]

def add_noise(clean_wav, noise_wavs, noise_snr=0):
    #use code from https://github.com/roudimit/whisper-flamingo/blob/main/utils.py
    clean_wav = clean_wav.astype(np.float32)
    noise_wav = select_noise(noise_wavs)
    if type(noise_snr) == int or type(noise_snr) == float:
        snr = noise_snr
    elif type(noise_snr) == tuple:
        snr = np.random.randint(noise_snr[0], noise_snr[1]+1)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10**(snr/20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav

    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
            reduction_rate = max_int16 / mixed.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    mixed = mixed.astype(np.int16)
    return mixed
    
def load_audio_with_noise(root_dir,wav_path,processor,SNR=0):
    audio_path,noise_fn=  wav_path[0],wav_path[1]
    audio_path = root_dir + audio_path.replace(".mp4",".wav")
    sampling_rate, audio_array = wavfile.read(audio_path)
    audio_array = add_noise(audio_array, [noise_fn], noise_snr=SNR).flatten().astype(np.float32) / 32768.0
    #wavfile读取的音频除去32768和librosa一样

    # wavfile.write('vis/'+os.path.basename(audio_path), sampling_rate, (audio_array * 32768).astype(np.int16))

    mel_spectrogram=processor.feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]
    return mel_spectrogram,audio_array

def load_video(root_dir,vidname,image_mean = 0.421,image_std = 0.165):
    vidname=  vidname[0]
    vr = video_loader(root_dir + vidname) 
    buffer = vr.get_batch(range(len(vr))).asnumpy()
    #水平扩增
    # buffer = video.horizontal_flip(buffer,prob=0.5)
    video_length = len(buffer)

    weights = np.array([0.1140, 0.5870, 0.2989])

    buffer = np.expand_dims(np.dot(buffer[..., :3], weights).astype(np.uint8),axis=-1)
    # 如果帧少于750，补充黑色帧
    original_length = len(buffer)
    if original_length < 750:
        height, width, channels = buffer.shape[1:4]  # 假设形状为 (帧数, 高, 宽, 通道数)
        num_blank_frames = 750 - original_length
        blank_frames = np.zeros((num_blank_frames, height, width, channels), dtype=buffer.dtype)
        buffer = np.concatenate((buffer, blank_frames), axis=0)
    buffer = buffer/ 255.0
    buffer = (buffer - image_mean) / image_std
    buffer = buffer.transpose(1, 2, 3, 0)  # Change shape to  H x W x C x T
    return (torch.FloatTensor(buffer),video_length)

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="en", task="transcribe")

def get_audio_video(root_dir,vidname,processor,image_mean = 0.421,image_std = 0.165,with_noise=False,SNR=0):
    if processor is None:
        processor = whisper_processor
        
    if not with_noise:
        audio_mel,audio_wav = load_audio(root_dir,vidname,processor)
    else:
        audio_mel,audio_wav = load_audio_with_noise(root_dir,vidname,processor,SNR=SNR)

    video,length = load_video(root_dir,vidname,image_mean =image_mean,image_std =image_std)

    audio_mel = processor.feature_extractor.pad([{'input_features':audio_mel}], return_tensors="pt")['input_features']
    video = torch.stack([ video ],dim=0)
    return audio_mel,video,length,audio_wav


def get_local_wav(x):
    base_name = os.path.basename(x)
    folder = base_name.split("_")[0]
    id = base_name.split("_")[-1].replace(".wav", "_00.mp4")
    return os.path.join("lrs2/lip/mvlrs_v1/main/", folder, id)

def process_test_file(test_file= "data/mix_2_spk_tt.scp"):
    """读取测试文件并生成音频文件路径及对应的信噪比数据"""
    out_files = []
    with open(test_file, "r") as f:
        test_files = f.readlines()
    for x in test_files:
        x = x.strip()
        if not x:  # 跳过空行
            continue
        # 获取源文件、噪声文件和信噪比
        source = get_local_wav(x.split(" ")[0])
        noise = get_local_wav(x.split(" ")[2])
        snr = float(x.split(" ")[-1])
        out_files.append([source, noise, snr])
        out_files.append([noise, source, -snr])
    return out_files

