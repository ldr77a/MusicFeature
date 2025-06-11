import os
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import joblib
import warnings
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import tempfile
import shutil

# Librosa 경고 메시지 필터링
warnings.filterwarnings("ignore", message="n_fft=.*is too large for input signal of length.*")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 실제 운영시에는 보안이 강한 키로 변경

# 업로드 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB 제한

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/results', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(y, sr, start_time, end_time):
    """
    주어진 오디오 세그먼트에서 음악 특징을 추출합니다.
    """
    # 시간 인덱스를 샘플 인덱스로 변환
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    
    # 세그먼트 추출
    y_segment = y[start_idx:end_idx]
    
    # 신호의 길이에 맞게 적절한 n_fft 값 설정
    signal_length = len(y_segment)
    
    if signal_length < 1024:
        n_fft = 2 ** int(np.log2(signal_length))
        n_fft = max(32, n_fft)
        hop_length = n_fft // 4
    else:
        n_fft = 1024
        hop_length = 256
    
    # 특징들을 저장할 딕셔너리
    features = {}
    
    # 1. 다양한 Chroma 특징들
    chroma_cens = librosa.feature.chroma_cens(y=y_segment, sr=sr, hop_length=hop_length)
    chroma_stft = librosa.feature.chroma_stft(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma_cqt = librosa.feature.chroma_cqt(y=y_segment, sr=sr, hop_length=hop_length)
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, note in enumerate(note_names):
        features[f'{note}_CENS_mean'] = np.mean(chroma_cens[i])
        features[f'{note}_CENS_std'] = np.std(chroma_cens[i])
        features[f'{note}_STFT_mean'] = np.mean(chroma_stft[i])
        features[f'{note}_STFT_std'] = np.std(chroma_stft[i])
        features[f'{note}_CQT_mean'] = np.mean(chroma_cqt[i])
        features[f'{note}_CQT_std'] = np.std(chroma_cqt[i])
    
    # 2. 확장된 MFCC
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    for i in range(n_mfcc):
        features[f'MFCC_{i}_mean'] = np.mean(mfcc[i])
        features[f'MFCC_{i}_std'] = np.std(mfcc[i])
        
        if i < 10:
            features[f'MFCC_{i}_kurtosis'] = kurtosis(mfcc[i], fisher=True, nan_policy='omit')
            features[f'MFCC_{i}_skewness'] = skew(mfcc[i], nan_policy='omit')
        
        features[f'MFCC_delta_{i}_mean'] = np.mean(mfcc_delta[i])
        features[f'MFCC_delta_{i}_std'] = np.std(mfcc_delta[i])
        
        if i < 10:
            features[f'MFCC_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
            features[f'MFCC_delta2_{i}_std'] = np.std(mfcc_delta2[i])
    
    # 3. Tonnetz
    y_harmonic = librosa.effects.harmonic(y_segment)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    
    tonnetz_names = ['tonnetz_0', 'tonnetz_1', 'tonnetz_2', 'tonnetz_3', 'tonnetz_4', 'tonnetz_5']
    for i, name in enumerate(tonnetz_names):
        features[f'{name}_mean'] = np.mean(tonnetz[i])
        features[f'{name}_std'] = np.std(tonnetz[i])
    
    # 4. 스펙트럼 특성들
    spectral_centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for i in range(7):
        features[f'contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        features[f'contrast_{i}_std'] = np.std(spectral_contrast[i])
    
    spectral_flatness = librosa.feature.spectral_flatness(y=y_segment, n_fft=n_fft, hop_length=hop_length)
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_segment, hop_length=hop_length)
    features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
    features['zero_crossing_rate_std'] = np.std(zero_crossing_rate)
    
    # 5. 리듬 특성들
    tempo, _ = librosa.beat.beat_track(y=y_segment, sr=sr, hop_length=hop_length)
    if hasattr(tempo, 'item'):
        tempo = tempo.item()
    else:
        tempo = float(tempo)
    features['tempo'] = tempo
    
    y_perc = librosa.effects.percussive(y_segment)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    ac = librosa.autocorrelate(onset_env, max_size=sr // hop_length // 2)
    ac = librosa.util.normalize(ac, norm=np.inf)
    
    peaks = librosa.util.peak_pick(ac, pre_max=10, post_max=10, pre_avg=10, post_avg=10, delta=0.5, wait=10)
    
    if len(peaks) > 0:
        features['rhythm_periodicity'] = len(peaks)
        features['rhythm_strength'] = np.mean(ac[peaks])
    else:
        features['rhythm_periodicity'] = 0
        features['rhythm_strength'] = 0
    
    features['onset_strength_mean'] = np.mean(onset_env)
    features['onset_strength_std'] = np.std(onset_env)
    
    # 6. Band Energy Ratio
    D = np.abs(librosa.stft(y_segment, n_fft=n_fft, hop_length=hop_length))
    
    n_bands = 3
    band_size = D.shape[0] // n_bands
    
    band_energy = []
    for i in range(n_bands):
        start_bin = i * band_size
        end_bin = min((i + 1) * band_size, D.shape[0])
        band_energy.append(np.sum(D[start_bin:end_bin, :] ** 2))
    
    total_energy = np.sum(band_energy)
    if total_energy > 0:
        for i in range(n_bands):
            features[f'band_energy_ratio_{i}'] = band_energy[i] / total_energy
    else:
        for i in range(n_bands):
            features[f'band_energy_ratio_{i}'] = 0
    
    # 7. RMS 에너지
    rms = librosa.feature.rms(y=y_segment, hop_length=hop_length)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # 8. 하모닉/퍼커시브 성분 비율
    y_harmonic = librosa.effects.harmonic(y_segment)
    y_percussive = librosa.effects.percussive(y_segment)
    
    harmonic_energy = np.sum(y_harmonic ** 2)
    percussive_energy = np.sum(y_percussive ** 2)
    total_energy = harmonic_energy + percussive_energy
    
    if total_energy > 0:
        features['harmonic_ratio'] = harmonic_energy / total_energy
        features['percussive_ratio'] = percussive_energy / total_energy
    else:
        features['harmonic_ratio'] = 0
        features['percussive_ratio'] = 0
    
    features['start_time'] = start_time
    features['end_time'] = end_time
    
    # NaN 값을 0으로 대체
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0
    
    return features

def detect_strong_beats(file_path):
    """강한 비트 시점 감지"""
    try:
        # 오디오 로드
        y, sr = librosa.load(file_path)
        
        # Harmonic-Percussive 분리
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # 비트 추적
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        if len(beat_times) == 0:
            print("No beats detected")
            return []
        
        # Onset strength 계산
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        beat_onset_frames = librosa.time_to_frames(beat_times, sr=sr)
        beat_onset_frames = np.minimum(beat_onset_frames, len(onset_env) - 1)
        beat_strengths = onset_env[beat_onset_frames]
        
        # 곡의 특성에 맞는 강한 비트 선택 (70th percentile)
        threshold = np.percentile(beat_strengths, 70)
        strong_beat_indices = np.where(beat_strengths > threshold)[0]
        strong_beat_times = beat_times[strong_beat_indices]
        
        print(f"총 비트: {len(beat_times)}, 강한 비트: {len(strong_beat_times)}, 임계값: {threshold:.4f}")
        print(f"비트 세기 범위: {np.min(beat_strengths):.4f} ~ {np.max(beat_strengths):.4f}")
        print(f"평균 비트 간격: {np.mean(np.diff(strong_beat_times)):.2f}초" if len(strong_beat_times) > 1 else "")
        
        # 동적 임계값 조정: 너무 적으면 60th percentile, 그래도 적으면 50th percentile 사용
        if len(strong_beat_times) < max(5, len(beat_times) * 0.1):  # 전체 비트의 10% 미만이면
            print("⚠️ 70th percentile 비트가 너무 적음, 60th percentile로 재시도")
            threshold = np.percentile(beat_strengths, 60)
            strong_beat_indices = np.where(beat_strengths > threshold)[0]
            strong_beat_times = beat_times[strong_beat_indices]
            
            if len(strong_beat_times) < max(5, len(beat_times) * 0.15):  # 여전히 적으면
                print("⚠️ 60th percentile도 너무 적음, 50th percentile 사용")
                threshold = np.percentile(beat_strengths, 50)
                strong_beat_indices = np.where(beat_strengths > threshold)[0]
                strong_beat_times = beat_times[strong_beat_indices]
        
        print(f"최종 강한 비트: {len(strong_beat_times)}개 (전체의 {len(strong_beat_times)/len(beat_times)*100:.1f}%)")
        
        return strong_beat_times.tolist()
        
    except Exception as e:
        print(f"비트 분석 에러: {e}")
        return []

def load_trained_model_and_scaler(model_dir="Model"):
    """
    학습된 SMOTE 모델과 스케일러를 로드합니다.
    """
    model_path = os.path.join(model_dir, 'smote_model2.pkl')
    scaler_path = os.path.join(model_dir, 'smote_scaler2.pkl')
    features_path = os.path.join(model_dir, 'smote_features.txt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 파일이 존재하지 않습니다: {scaler_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"특징 목록 파일이 존재하지 않습니다: {features_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    emotion_mapping = {
        1: 'Sad',
        2: 'Annoying', 
        3: 'Calme',
        4: 'Amusing'
    }
    
    return model, scaler, feature_names, emotion_mapping

def apply_emotion_threshold(probabilities, thresholds):
    """
    감정별 확률 임계값을 적용하여 더 균형잡힌 예측을 수행합니다.
    """
    original_emotion_id = np.argmax(probabilities) + 1
    
    adjusted_probs = probabilities.copy()
    
    if 4 in thresholds:
        if probabilities[3] < thresholds[4]:
            adjusted_probs[3] *= 0.55
    
    for emotion_id, threshold in thresholds.items():
        if emotion_id <= 3:
            if probabilities[emotion_id-1] >= threshold:
                adjusted_probs[emotion_id-1] *= 1.2
    
    adjusted_emotion_id = np.argmax(adjusted_probs) + 1
    confidence = np.max(adjusted_probs)
    
    return adjusted_emotion_id, confidence, original_emotion_id

def process_mp3_file(file_path, model, scaler, feature_names, emotion_mapping, segment_duration=2.0):
    """
    MP3 파일을 지정된 간격으로 처리하여 감정 분류를 수행합니다.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        raise ValueError(f"오디오 파일 로드 실패: {e}")
    
    duration = librosa.get_duration(y=y, sr=sr)
    
    emotion_thresholds = {
        1: 0.4,  # Sad
        2: 0.4,  # Annoying  
        3: 0.4,  # Calme
        4: 0.6   # Amusing
    }
    
    results = []
    start_times = np.arange(0, duration, segment_duration)
    
    for i, start_time in enumerate(start_times):
        end_time = min(start_time + segment_duration, duration)
        
        try:
            features = extract_features(y, sr, start_time, end_time)
        except Exception as e:
            continue
        
        selected_features = []
        for feature_name in feature_names:
            if feature_name in features:
                selected_features.append(features[feature_name])
            else:
                selected_features.append(0)
        
        X = np.array(selected_features).reshape(1, -1)
        
        try:
            X_scaled = scaler.transform(X)
            emotion_prob = model.predict_proba(X_scaled)[0]
            
            emotion_id, confidence, original_emotion_id = apply_emotion_threshold(
                emotion_prob, emotion_thresholds)
            emotion_name = emotion_mapping.get(emotion_id, 'Unknown')
            
        except Exception as e:
            continue
        
        result = {
            'segment_id': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'emotion_id': emotion_id,
            'emotion': emotion_name,
            'confidence': confidence,
        }
        
        model_classes = model.classes_
        for idx, class_label in enumerate(model_classes):
            emotion_name_for_prob = emotion_mapping.get(class_label, f'Unknown_{class_label}')
            result[f'probability_{emotion_name_for_prob}'] = emotion_prob[idx]
        
        results.append(result)
    
    df = pd.DataFrame(results)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('파일이 선택되지 않았습니다.')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('파일이 선택되지 않았습니다.')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # 모델 로드
            model, scaler, feature_names, emotion_mapping = load_trained_model_and_scaler()
            
            # 감정 분석 수행
            results_df = process_mp3_file(file_path, model, scaler, feature_names, emotion_mapping)
            
            # 강한 비트 분석 수행
            strong_beats = detect_strong_beats(file_path)
            
            # 결과를 JSON으로 변환
            results_json = results_df.to_dict('records')
            
            # 결과 저장
            result_filename = f"{os.path.splitext(filename)[0]}_emotion_analysis.json"
            result_path = os.path.join('static/results', result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, ensure_ascii=False, indent=2)
            
            # 업로드된 파일을 static 폴더로 복사 (웹에서 접근 가능하도록)
            static_audio_path = os.path.join('static', filename)
            shutil.copy2(file_path, static_audio_path)
            
            return render_template('result.html', 
                                 audio_file=filename,
                                 results=results_json,
                                 strong_beats=strong_beats,
                                 total_duration=results_df['end_time'].max() if len(results_df) > 0 else 0)
        
        except Exception as e:
            flash(f'분석 중 오류가 발생했습니다: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('MP3 파일만 업로드 가능합니다.')
        return redirect(url_for('index'))

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/emotion_images/<path:filename>')
def emotion_images(filename):
    return send_from_directory('Emotion_png', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 