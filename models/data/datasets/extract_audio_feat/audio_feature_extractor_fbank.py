#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AVIS ↔ Lavish : WAV → fbank(192×192) 저장 스크립트
폴더 구조 / print 로그는 AVIS 코드 그대로 유지.
"""

import os, contextlib, wave, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch, torchaudio                    
from tqdm import tqdm                    
import matplotlib.pyplot as plt        # 추가
import matplotlib.cm as cm             # 컬러맵 지정용 (선택)

# ─────────────── Lavish fbank 설정 ────────────────
N_MEL_BINS    = 192
TARGET_FRAMES = 192            # 10 ms hop → 1.92 s
# !!! 데이터셋(train) 전체에서 계산
NORM_MEAN     = -5.8145327568  # placeholder
NORM_STD      =  4.3201255798  # placeholder

# ─────────────── HELPER: WAV → (num_secs,1,192,192) ───────────────
def wav_to_fbank_segments(wav_path: str, num_secs: int) -> np.ndarray:
    """WAV 전체를 1초 단위(num_secs개)로 잘라 fbank (192 x 192)로 변환"""
    waveform, sr = torchaudio.load(wav_path)       # mono (1,T)
    waveform -= waveform.mean()

    seg_len = sr                                   # 1 s
    need_len = num_secs * seg_len
    if waveform.size(1) < need_len:                # 부족하면 0-패딩
        waveform = torch.nn.functional.pad(
            waveform, (0, need_len - waveform.size(1)))

    feats = []
    for t in range(num_secs):
        seg = waveform[:, t*seg_len:(t+1)*seg_len]

        fbank = torchaudio.compliance.kaldi.fbank(
            seg, htk_compat=True, sample_frequency=sr,
            use_energy=False, window_type='hanning',
            num_mel_bins=N_MEL_BINS, dither=0.0, frame_shift=10
        )                                          # (T,192)

        # fbank = (fbank - NORM_MEAN) / (NORM_STD * 2)

        if fbank.size(0) < TARGET_FRAMES:
            fbank = torch.nn.functional.pad(
                fbank, (0, 0, 0, TARGET_FRAMES - fbank.size(0)))
        else:
            fbank = fbank[:TARGET_FRAMES]

        feats.append(fbank)

    feats = torch.stack(feats).unsqueeze(1)      # (N,1,192,192)
    return feats.numpy().astype(np.float32)


def get_audio_len(audio_file):
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        return int(f.getnframes() / float(f.getframerate()))

# def wav_to_fbank(wav_path):
#     wav, sr = torchaudio.load(wav_path)
#     wav -= wav.mean()
#     fbank = torchaudio.compliance.kaldi.fbank(
#         wav, sample_frequency=sr, use_energy=False,
#         window_type='hanning', num_mel_bins=N_MEL_BINS,
#         dither=0.0, frame_shift=10
#     )
#     # 길이 보정
#     if fbank.size(0) < TARGET_FRAMES:
#         fbank = torch.nn.functional.pad(
#             fbank, (0, 0, 0, TARGET_FRAMES - fbank.size(0)))
#     else:
#         fbank = fbank[:TARGET_FRAMES]
#     return fbank.reshape(-1)        


# count = 0
# mean  = torch.zeros(1)
# M2    = torch.zeros(1)


# np.save("/home/diml/hddsdb/jin/AVIS/puppies/feat_mel/puppies.npy", wav_to_fbank_segments("/home/diml/hddsdb/jin/AVIS/puppies/wav/puppies.wav", 20))
# exit()

audio_root = "./datasets/"
# for subset in ["train", "val", "test"]:
for subset in ["test"]:
    print(f"{subset} ----------> ")

    audio_dir = os.path.join(audio_root, subset, "WAVAudios")
    save_dir  = os.path.join(audio_root, subset, "MELAudios")
    save_dir_img = os.path.join(audio_root, subset, "MELImgs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_img, exist_ok=True)

    lis = sorted(os.listdir(audio_dir))
    print(len(lis))

    for idx, fname in enumerate(tqdm(lis, ncols=80)):
        if not fname.endswith(".wav"):    
            continue

        # outfile = os.path.join(save_dir, fname[:-4] + '.npy')
        # if os.path.exists(outfile):
        #     print(f"\nProcessing: {idx+1}/{len(lis)} ----> {fname[:-4]}.npy is already exist!")
        #     continue

        audio_path = os.path.join(audio_dir, fname)
        num_secs   = len(os.listdir(os.path.join(
                         audio_root, subset, "JPEGImages", fname[:-4])))
        
        # for x in wav_to_fbank(audio_path):
        #     count += 1
        #     delta  = x - mean
        #     mean  += delta / count
        #     M2    += delta * (x - mean)

        # ── 여기 한 줄만 변경 ───────────────────────────────
        input_batch = wav_to_fbank_segments(audio_path, num_secs)  # (N,1,192,192)
        # ───────────────────────────────────────────────────
        
        img_dir = os.path.join(save_dir_img, fname[:-4])
        os.makedirs(img_dir, exist_ok=True)

        # np.save(outfile, input_batch)
        for t in range(input_batch.shape[0]):
            spec = input_batch[t, 0]             # (192,192)  float32

            # 0~1 스케일 정규화 후 컬러맵 적용 (원한다면 회색조 그대로도 가능)
            # spec_norm = (spec - spec.min()) / (spec.ptp() + 1e-8)
            plt.imsave(
                os.path.join(img_dir, f"{fname[:-4]}_{t}.png"),
                spec,
                cmap="magma"      # "gray" , "viridis" 등 원하는 컬러맵
            )

        print(f"save info: {fname[:-4]} → {input_batch.shape[0]} PNG files")
        
        # print(f" save info: {fname[:-4]}.npy  --->  {input_batch.shape}")
    
    print("\n---------------------------------- end ----------------------------------\n")

# std = torch.sqrt(M2 / (count - 1))
# print(f"NEW_MEAN = {mean.item():.10f}")
# print(f"NEW_STD  = {std.item():.10f}")