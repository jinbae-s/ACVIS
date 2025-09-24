import os
import subprocess
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim
import contextlib
import wave

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 사용할 GPU 번호

FPS = 30

def get_audio_len(wav_path):
    with contextlib.closing(wave.open(wav_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return int(frames / float(rate))

def extract_wav_from_mp4(mp4_path, wav_path, sample_rate=44100):
    cmd = [
        'ffmpeg',
        '-y',                 # 기존 파일 덮어쓰기
        '-i', mp4_path,       # 입력 비디오
        '-vn',                # 비디오 스트림 제외
        '-ac', '2',           # stereo 
        '-ar', str(sample_rate),  # 샘플링 레이트
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

mp4_path   = '/home/diml/hddsdb/jin/AVIS/puppies/puppies.mp4'      
out_dir    = '/home/diml/hddsdb/jin/AVIS/puppies'           
os.makedirs(out_dir, exist_ok=True)

base_name  = os.path.splitext(os.path.basename(mp4_path))[0]
wav_path   = os.path.join(out_dir, base_name + '.wav')
npy_path   = os.path.join(out_dir, base_name + '.npy')

checkpoint_path = './vggish_model.ckpt'

print(f"🔊 Extracting audio → {wav_path}")
extract_wav_from_mp4(mp4_path, wav_path)

num_secs = get_audio_len(wav_path) * FPS
print(f"⏱ Audio length: {num_secs} sec")

examples = vggish_input.wavfile_to_examples(wav_path, num_secs)
expected_shape = (num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS)
np.testing.assert_equal(examples.shape, expected_shape)
print(f"✅ Input batch shape: {examples.shape}")

with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    feat_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    emb_tensor  = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    [emb_batch] = sess.run([emb_tensor], feed_dict={feat_tensor: examples})

np.save(npy_path, emb_batch)
print(f"Saved embeddings → {npy_path}, shape={emb_batch.shape}")
