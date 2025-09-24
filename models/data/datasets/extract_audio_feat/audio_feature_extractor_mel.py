import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # set gpu number
import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_slim
import contextlib
import wave


# get audio length
def get_audio_len(audio_file):
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        wav_length = int(frames / float(rate))
        return wav_length

# Paths to downloaded VGGish files.
checkpoint_path = './vggish_model.ckpt'
pca_params_path = './vggish_pca_params.npz'
freq = 1000
sr = 44100


audio_root = "/home/diml/hddsdb/jin/AVIS/datasets"
for subset in ["train", "val", "test"]:
    print("{} ----------> ".format(subset))

    audio_dir = os.path.join(audio_root, subset, "WAVAudios")
    save_dir = os.path.join(audio_root, subset, "MELAudios")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lis = sorted(os.listdir(audio_dir))
    len_data = len(lis)
    print(len_data)

    i = 0
    for n in range(len_data):
        i += 1
        # save file
        outfile = os.path.join(save_dir, lis[n][:-4] + '.npy')
        if os.path.exists(outfile):
            print("\nProcessing: ", i, " / ", len_data, " ----> ", lis[n][:-4] + '.npy', " is already exist! ")
            continue

        '''feature learning by VGG-net trained by audioset'''
        audio_index = os.path.join(audio_dir, lis[n]) # path of your audio files
        num_secs = len(os.listdir(os.path.join(audio_root, subset, "JPEGImages", lis[n][:-4])))


        input_batch = vggish_input.wavfile_to_examples(audio_index, num_secs)
        np.testing.assert_equal(
            input_batch.shape,
            [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])


        np.save(outfile, input_batch)
        print(" save info: ", lis[n][:-4] + '.npy', " ---> ", input_batch.shape)

        i += 1

    print("\n---------------------------------- end ----------------------------------\n")


