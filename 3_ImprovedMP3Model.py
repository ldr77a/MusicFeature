import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import joblib
import warnings
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import argparse

# Librosa 경고 메시지 필터링
warnings.filterwarnings("ignore", message="n_fft=.*is too large for input signal of length.*")

def extract_features(y, sr, start_time, end_time):
    """
    주어진 오디오 세그먼트에서 음악 특징을 추출합니다.
    
    Parameters:
    y (np.ndarray): 오디오 시계열
    sr (int): 샘플링 레이트
    start_time (float): 시작 시간 (초)
    end_time (float): 종료 시간 (초)
    
    Returns:
    dict: 추출된 특징들을 포함하는 사전
    """
    # 시간 인덱스를 샘플 인덱스로 변환
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    
    # 세그먼트 추출
    y_segment = y[start_idx:end_idx]
    
    # 신호의 길이에 맞게 적절한 n_fft 값 설정
    signal_length = len(y_segment)
    
    # n_fft를 신호 길이보다 작게 설정 (2의 거듭제곱 값으로)
    if signal_length < 1024:
        # 신호 길이보다 작은 2의 거듭제곱 값 찾기
        n_fft = 2 ** int(np.log2(signal_length))
        # 최소 n_fft 값이 32보다 작으면 32로 설정
        n_fft = max(32, n_fft)
        hop_length = n_fft // 4
    else:
        n_fft = 1024
        hop_length = 256
    
    # 특징들을 저장할 딕셔너리
    features = {}
    
    # 1. 다양한 Chroma 특징들
    
    # 1.1 Chroma Energy Normalized (CENS)
    chroma_cens = librosa.feature.chroma_cens(y=y_segment, sr=sr, hop_length=hop_length)
    
    # 1.2 기본 Chroma feature
    chroma_stft = librosa.feature.chroma_stft(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 1.3 Constant-Q chroma
    chroma_cqt = librosa.feature.chroma_cqt(y=y_segment, sr=sr, hop_length=hop_length)
    
    # 각 음계별 평균값과 표준편차 계산 (통계적 특성 확장)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, note in enumerate(note_names):
        # CENS
        features[f'{note}_CENS_mean'] = np.mean(chroma_cens[i])
        features[f'{note}_CENS_std'] = np.std(chroma_cens[i])
        
        # STFT
        features[f'{note}_STFT_mean'] = np.mean(chroma_stft[i])
        features[f'{note}_STFT_std'] = np.std(chroma_stft[i])
        
        # CQT
        features[f'{note}_CQT_mean'] = np.mean(chroma_cqt[i])
        features[f'{note}_CQT_std'] = np.std(chroma_cqt[i])
    
    # 2. 확장된 MFCC - 20개의 계수 + 델타 + 델타델타
    n_mfcc = 20  # MFCC 개수를 20개로 확장
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)  # 델타 MFCC
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)  # 델타델타 MFCC
    
    # 각 MFCC 계수별 평균, 표준편차, 첨도, 왜도 계산
    for i in range(n_mfcc):
        # 기본 MFCC
        features[f'MFCC_{i}_mean'] = np.mean(mfcc[i])
        features[f'MFCC_{i}_std'] = np.std(mfcc[i])
        
        # 첨도와 왜도는 통계적으로 유용한 특성이지만, 첫 10개 계수에만 적용하여 특징 수 제한
        if i < 10:
            features[f'MFCC_{i}_kurtosis'] = kurtosis(mfcc[i], fisher=True, nan_policy='omit')
            features[f'MFCC_{i}_skewness'] = skew(mfcc[i], nan_policy='omit')
        
        # 델타 MFCC (변화율)
        features[f'MFCC_delta_{i}_mean'] = np.mean(mfcc_delta[i])
        features[f'MFCC_delta_{i}_std'] = np.std(mfcc_delta[i])
        
        # 델타-델타 MFCC (가속도) - 첫 10개만 사용하여 특징 수 제한
        if i < 10:
            features[f'MFCC_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
            features[f'MFCC_delta2_{i}_std'] = np.std(mfcc_delta2[i])
    
    # 3. Tonnetz - 조성 센트로이드와 관련된 특징
    y_harmonic = librosa.effects.harmonic(y_segment)  # 하모닉 성분 추출
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    
    # 각 톤네츠 차원의 특성 추출 (6개 차원)
    tonnetz_names = ['tonnetz_0', 'tonnetz_1', 'tonnetz_2', 'tonnetz_3', 'tonnetz_4', 'tonnetz_5']
    for i, name in enumerate(tonnetz_names):
        features[f'{name}_mean'] = np.mean(tonnetz[i])
        features[f'{name}_std'] = np.std(tonnetz[i])
    
    # 4. 스펙트럼 특성들
    
    # 4.1 Spectral Centroid (평균과 표준편차)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    # 4.2 Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # 4.3 Spectral Contrast (7개 대역)
    spectral_contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    for i in range(7):
        features[f'contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        features[f'contrast_{i}_std'] = np.std(spectral_contrast[i])
    
    # 4.4 Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y_segment, n_fft=n_fft, hop_length=hop_length)
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)
    
    # 4.5 Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # 4.6 Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_segment, hop_length=hop_length)
    features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
    features['zero_crossing_rate_std'] = np.std(zero_crossing_rate)
    
    # 5. 리듬 특성들
    
    # 5.1 Tempo
    tempo, _ = librosa.beat.beat_track(y=y_segment, sr=sr, hop_length=hop_length)
    # NumPy 배열을 스칼라로 변환할 때 경고를 방지하기 위한 올바른 방법
    if hasattr(tempo, 'item'):
        tempo = tempo.item()  # NumPy 배열에서 단일 값 추출
    else:
        tempo = float(tempo)  # 다른 타입의 경우 단순 변환
    features['tempo'] = tempo
    
    # 5.2 리듬 특성: 시간축 Autocorrelation - 리듬 규칙성 측정
    # 리듬의 자기상관 함수는 반복적인 리듬 패턴을 찾는데 유용합니다
    y_perc = librosa.effects.percussive(y_segment)  # 타악기(퍼커시브) 성분 추출
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    ac = librosa.autocorrelate(onset_env, max_size=sr // hop_length // 2)
    ac = librosa.util.normalize(ac, norm=np.inf)
    
    # 자기상관 함수에서 피크 추출 (리듬 주기성 측정)
    peaks = librosa.util.peak_pick(ac, pre_max=10, post_max=10, pre_avg=10, post_avg=10, delta=0.5, wait=10)
    
    if len(peaks) > 0:
        features['rhythm_periodicity'] = len(peaks)  # 피크 수 (리듬 복잡성 지표)
        features['rhythm_strength'] = np.mean(ac[peaks])  # 피크 강도 평균 (리듬 강조 정도)
    else:
        features['rhythm_periodicity'] = 0
        features['rhythm_strength'] = 0
    
    # 5.3 리듬 강도 통계
    features['onset_strength_mean'] = np.mean(onset_env)
    features['onset_strength_std'] = np.std(onset_env)
    
    # 6. Band Energy Ratio
    # 주파수 대역별 에너지 분포 - 저주파/중간주파/고주파 에너지의 상대적 비율
    # 오디오 신호의 에너지가 어떤 주파수 대역에 집중되어 있는지 파악
    D = np.abs(librosa.stft(y_segment, n_fft=n_fft, hop_length=hop_length))
    
    # 주파수 빈을 3개 대역으로 나누기
    n_bands = 3
    band_size = D.shape[0] // n_bands
    
    # 각 주파수 대역의 에너지 계산
    band_energy = []
    for i in range(n_bands):
        start_bin = i * band_size
        end_bin = min((i + 1) * band_size, D.shape[0])
        band_energy.append(np.sum(D[start_bin:end_bin, :] ** 2))
    
    # 전체 에너지로 정규화하여 비율 계산
    total_energy = np.sum(band_energy)
    if total_energy > 0:
        for i in range(n_bands):
            features[f'band_energy_ratio_{i}'] = band_energy[i] / total_energy
    else:
        for i in range(n_bands):
            features[f'band_energy_ratio_{i}'] = 0
    
    # 7. RMS 에너지 - 음량과 관련
    rms = librosa.feature.rms(y=y_segment, hop_length=hop_length)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # 8. 하모닉/퍼커시브 성분 비율
    y_harmonic = librosa.effects.harmonic(y_segment)
    y_percussive = librosa.effects.percussive(y_segment)
    
    # 하모닉 에너지와 퍼커시브 에너지 계산
    harmonic_energy = np.sum(y_harmonic ** 2)
    percussive_energy = np.sum(y_percussive ** 2)
    total_energy = harmonic_energy + percussive_energy
    
    if total_energy > 0:
        features['harmonic_ratio'] = harmonic_energy / total_energy
        features['percussive_ratio'] = percussive_energy / total_energy
    else:
        features['harmonic_ratio'] = 0
        features['percussive_ratio'] = 0
    
    # 시작 및 종료 시간 추가
    features['start_time'] = start_time
    features['end_time'] = end_time
    
    # NaN 값을 0으로 대체
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0
    
    return features

def load_trained_model_and_scaler(model_dir="Model"):
    """
    학습된 SMOTE 모델과 스케일러를 로드합니다.
    
    Args:
        model_dir (str): 모델이 저장된 디렉토리
        
    Returns:
        tuple: (모델, 스케일러, 사용된 특징 목록, 감정 레이블 매핑)
    """
    print(f"'{model_dir}' 폴더에서 학습된 모델을 로드 중...")
    
    # 모델 파일 경로 (새로운 모델 우선 시도)
    model_path = os.path.join(model_dir, 'smote_model2.pkl')
    scaler_path = os.path.join(model_dir, 'smote_scaler2.pkl')
    
    # 새 모델이 없으면 기존 모델 사용
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'smote_model2.pkl')
        scaler_path = os.path.join(model_dir, 'smote_scaler2.pkl')
    
    features_path = os.path.join(model_dir, 'smote_features.txt')
    
    # 파일 존재 여부 확인
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 파일이 존재하지 않습니다: {scaler_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"특징 목록 파일이 존재하지 않습니다: {features_path}")
    
    # 모델과 스케일러 로드
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 사용된 특징 목록 로드
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # 감정 레이블 매핑 (emotion_id -> emotion_name)
    emotion_mapping = {
        1: 'Sad',
        2: 'Annoying', 
        3: 'Calme',
        4: 'Amusing'
    }
    
    print(f"모델 로드 완료. 사용할 특징 수: {len(feature_names)}")
    print("감정 레이블 매핑:")
    for emotion_id, emotion_name in emotion_mapping.items():
        print(f"  {emotion_id}: {emotion_name}")
    
    return model, scaler, feature_names, emotion_mapping

def apply_emotion_threshold(probabilities, thresholds):
    """
    감정별 확률 임계값을 적용하여 더 균형잡힌 예측을 수행합니다.
    
    Args:
        probabilities (np.array): 4개 감정의 확률 [Sad, Annoying, Calme, Amusing]
        thresholds (dict): 감정별 임계값 {emotion_id: threshold}
        
    Returns:
        tuple: (adjusted_emotion_id, confidence, original_emotion_id)
    """
    # 원본 예측
    original_emotion_id = np.argmax(probabilities) + 1
    
    # 임계값 적용
    adjusted_probs = probabilities.copy()
    
    # Amusing(4)의 임계값을 높여서 과도한 분류 방지
    if 4 in thresholds:
        if probabilities[3] < thresholds[4]:  # Amusing이 임계값보다 낮으면
            adjusted_probs[3] *= 0.55
    
    # 다른 감정들의 임계값 적용
    for emotion_id, threshold in thresholds.items():
        if emotion_id <= 3:  # Sad(1), Annoying(2), Calme(3)
            if probabilities[emotion_id-1] >= threshold:
                adjusted_probs[emotion_id-1] *= 1.2  # 확률을 증가
    
    # 조정된 예측
    adjusted_emotion_id = np.argmax(adjusted_probs) + 1
    confidence = np.max(adjusted_probs)
    
    return adjusted_emotion_id, confidence, original_emotion_id

def process_mp3_file(file_path, model, scaler, feature_names, emotion_mapping, 
                    segment_duration=5.0, use_threshold=True):
    """
    MP3 파일을 지정된 간격으로 처리하여 감정 분류를 수행합니다.
    
    Args:
        file_path (str): MP3 파일 경로
        model: 학습된 모델
        scaler: 학습된 스케일러
        feature_names (list): 사용할 특징 이름 목록
        emotion_mapping (dict): 감정 ID -> 감정 이름 매핑
        segment_duration (float): 세그먼트 지속 시간 (초)
        use_threshold (bool): 확률 임계값 적용 여부
        
    Returns:
        pd.DataFrame: 감정 분류 결과를 포함하는 데이터프레임
    """
    print(f"MP3 파일 처리 중: {file_path}")
    print(f"세그먼트 길이: {segment_duration}초")
    print(f"확률 임계값 적용: {'예' if use_threshold else '아니오'}")
    
    # 오디오 로드
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        raise ValueError(f"오디오 파일 로드 실패: {e}")
    
    # 파일 지속 시간 확인
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"파일 지속 시간: {duration:.2f}초")
    
    # 확률 임계값 설정 (Amusing 과도 분류 방지)
    emotion_thresholds = {
        1: 0.4,  # Sad
        2: 0.4,  # Annoying  
        3: 0.4,  # Calme
        4: 0.6   # Amusing - 높은 임계값으로 과도한 분류 방지
    }
    
    # 결과를 저장할 리스트
    results = []
    
    # 지정된 간격으로 세그먼트 생성
    start_times = np.arange(0, duration, segment_duration)
    print(f"총 {len(start_times)}개의 세그먼트를 처리합니다.")
    
    # 각 세그먼트에 대해 처리
    for i, start_time in enumerate(tqdm(start_times, desc="세그먼트 처리")):
        end_time = min(start_time + segment_duration, duration)
        
        # 특징 추출
        try:
            features = extract_features(y, sr, start_time, end_time)
        except Exception as e:
            print(f"경고: 세그먼트 {i+1} 특징 추출 실패: {e}")
            continue
        
        # 모델에 필요한 특징만 선택
        selected_features = []
        missing_features = []
        
        for feature_name in feature_names:
            if feature_name in features:
                selected_features.append(features[feature_name])
            else:
                selected_features.append(0)  # 누락된 특징은 0으로 대체
                missing_features.append(feature_name)
        
        if missing_features and i == 0:  # 첫 번째 세그먼트에서만 경고 출력
            print(f"경고: 누락된 특징들 (처음 5개만 표시): {missing_features[:5]}...")
        
        # 특징 벡터를 2D 배열로 변환 (1개 샘플)
        X = np.array(selected_features).reshape(1, -1)
        
        # 스케일링 적용
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            print(f"경고: 세그먼트 {i+1} 스케일링 실패: {e}")
            continue
        
        # 감정 예측
        try:
            emotion_prob = model.predict_proba(X_scaled)[0]
            
            if use_threshold:
                # 임계값 적용
                emotion_id, confidence, original_emotion_id = apply_emotion_threshold(
                    emotion_prob, emotion_thresholds)
                emotion_name = emotion_mapping.get(emotion_id, 'Unknown')
                original_emotion_name = emotion_mapping.get(original_emotion_id, 'Unknown')
            else:
                # 기본 예측
                emotion_id = model.predict(X_scaled)[0]
                confidence = np.max(emotion_prob)
                emotion_name = emotion_mapping.get(emotion_id, 'Unknown')
                original_emotion_id = emotion_id
                original_emotion_name = emotion_name
                
        except Exception as e:
            print(f"경고: 세그먼트 {i+1} 예측 실패: {e}")
            continue
        
        # 결과 저장
        result = {
            'segment_id': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'emotion_id': emotion_id,
            'emotion': emotion_name,
            'confidence': confidence,
        }
        
        # model.classes_를 기반으로 정확한 확률 매핑
        model_classes = model.classes_  # 예: [1, 2, 3, 4]
        for idx, class_label in enumerate(model_classes):
            emotion_name_for_prob = emotion_mapping.get(class_label, f'Unknown_{class_label}')
            result[f'probability_{emotion_name_for_prob}'] = emotion_prob[idx]
        
        # 임계값 적용시 원본 예측도 저장
        if use_threshold:
            result['original_emotion_id'] = original_emotion_id
            result['original_emotion'] = original_emotion_name
            
        results.append(result)
    
    # 결과를 데이터프레임으로 변환
    df = pd.DataFrame(results)
    
    print(f"처리 완료. 총 {len(df)}개의 세그먼트가 분석되었습니다.")
    
    return df

def get_mp3_file_interactive():
    """
    사용자가 대화형으로 MP3 파일을 선택할 수 있도록 합니다.
    
    Returns:
        str: 선택된 MP3 파일 경로
    """
    print("🎵 MP3 파일 선택 🎵")
    print("=" * 50)
    
    # TestMusic 폴더 확인
    test_music_folder = "TestMusic"
    if os.path.isdir(test_music_folder):
        mp3_files = [f for f in os.listdir(test_music_folder) if f.lower().endswith('.mp3')]
        
        if mp3_files:
            print(f"\n📁 '{test_music_folder}' 폴더의 MP3 파일 목록:")
            for i, file in enumerate(mp3_files, 1):
                print(f"  {i}. {file}")
            
            print(f"\n선택 옵션:")
            print(f"  1-{len(mp3_files)}: 위 목록에서 번호 선택")
            print(f"  0: 직접 파일 경로 입력")
            
            while True:
                try:
                    choice = input(f"\n선택하세요 (0-{len(mp3_files)}): ").strip()
                    
                    if choice == '0':
                        # 직접 파일 경로 입력
                        file_path = input("MP3 파일의 전체 경로를 입력하세요: ").strip()
                        if os.path.exists(file_path) and file_path.lower().endswith('.mp3'):
                            return file_path
                        else:
                            print("❌ 파일이 존재하지 않거나 MP3 파일이 아닙니다. 다시 입력해주세요.")
                            continue
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(mp3_files):
                        selected_file = os.path.join(test_music_folder, mp3_files[choice_num - 1])
                        return selected_file
                    else:
                        print(f"❌ 1-{len(mp3_files)} 또는 0을 입력해주세요.")
                
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
        else:
            print(f"\n📁 '{test_music_folder}' 폴더에 MP3 파일이 없습니다.")
    else:
        print(f"\n📁 '{test_music_folder}' 폴더가 없습니다.")
    
    # TestMusic 폴더에 파일이 없거나 폴더가 없는 경우
    print("\n직접 MP3 파일 경로를 입력해주세요.")
    while True:
        file_path = input("MP3 파일의 전체 경로를 입력하세요: ").strip()
        if file_path == '':
            print("❌ 파일 경로를 입력해주세요.")
            continue
        
        if os.path.exists(file_path) and file_path.lower().endswith('.mp3'):
            return file_path
        else:
            print("❌ 파일이 존재하지 않거나 MP3 파일이 아닙니다. 다시 입력해주세요.")



def smooth_predictions(df, emotion_mapping, window_size=3):
    """
    이동 평균 필터를 사용하여 감정 예측을 평활화합니다.
    
    Args:
        df (pd.DataFrame): 원본 예측 결과 데이터프레임
        emotion_mapping (dict): 감정 ID -> 감정 이름 매핑
        window_size (int): 평균을 계산할 창 크기 (홀수 권장)
        
    Returns:
        pd.DataFrame: 평활화된 예측 결과가 추가된 데이터프레임
    """
    print(f"\n결과 평활화 중... (창 크기: {window_size})")
    
    # 확률 컬럼 목록 생성
    prob_cols = [col for col in df.columns if col.startswith('probability_')]
    
    if not prob_cols:
        print("경고: 확률 컬럼이 없어 평활화를 건너뜁니다.")
        return df
    
    # 확률 값에 이동 평균 적용
    # center=True는 현재 포인트를 중심으로 창을 설정
    smoothed_probs = df[prob_cols].rolling(window=window_size, center=True, min_periods=1).mean()
    
    # 평활화된 확률을 기반으로 새로운 감정 예측
    # df.columns.get_loc()를 사용하여 클래스 ID를 동적으로 찾음
    emotion_mapping_rev = {v: k for k, v in emotion_mapping.items()}
    class_labels = [emotion_mapping_rev[col.replace('probability_', '')] for col in smoothed_probs.columns]
    
    smoothed_emotion_indices = np.argmax(smoothed_probs.values, axis=1)
    df['smoothed_emotion_id'] = np.array(class_labels)[smoothed_emotion_indices]
    df['smoothed_emotion'] = df['smoothed_emotion_id'].map(emotion_mapping)
    
    print("평활화 완료.")
    return df

def save_results(df, output_path, use_threshold=False):
    """
    분석 결과를 CSV 파일로 저장합니다.
    
    Args:
        df (pd.DataFrame): 분석 결과 데이터프레임
        output_path (str): 출력 파일 경로
        use_threshold (bool): 확률 임계값 적용 여부
    """
    # 결과 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"결과가 저장되었습니다: {output_path}")
    
    # 감정별 분포 출력
    print("\n📊 감정 분포:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
    
    # 임계값 적용시 원본과 비교
    if use_threshold and 'original_emotion' in df.columns:
        print("\n🔄 임계값 적용 전후 비교:")
        original_counts = df['original_emotion'].value_counts()
        for emotion in emotion_counts.index:
            original_count = original_counts.get(emotion, 0)
            adjusted_count = emotion_counts[emotion]
            change = adjusted_count - original_count
            print(f"  {emotion}: {original_count} → {adjusted_count} ({change:+d})")
    
    # 평균 신뢰도 출력
    avg_confidence = df['confidence'].mean()
    print(f"\n📈 평균 신뢰도: {avg_confidence:.3f}")

def main():
    parser = argparse.ArgumentParser(description='MP3 파일을 지정된 간격으로 감정 분류 (개선된 버전)')
    parser.add_argument('--input', '-i', help='입력 MP3 파일 경로 (선택사항, 없으면 대화형 선택)')
    parser.add_argument('--output', '-o', help='출력 CSV 파일 경로 (선택사항)')
    parser.add_argument('--model_dir', '-m', default='Model', help='모델 디렉토리 경로')
    parser.add_argument('--segment', '-s', type=float, default=5.0, help='세그먼트 길이 (초)')
    parser.add_argument('--no-threshold', action='store_true', help='확률 임계값 적용 안함')
    
    args = parser.parse_args()
    
    # 입력 파일 경로
    input_file = args.input
    if not input_file:
        # 대화형으로 파일 선택
        input_file = get_mp3_file_interactive()
    
    # 분석 설정 (자동 설정: 2초 세그먼트, 임계값 적용)
    if len(sys.argv) == 1:  # 인수 없이 실행된 경우
        segment_duration = 2.0
        use_threshold = True
        print(f"\n✅ 자동 설정 적용:")
        print(f"   세그먼트 길이: {segment_duration}초")
        print(f"   확률 임계값 적용: {'예' if use_threshold else '아니오'}")
    else:
        segment_duration = args.segment
        use_threshold = not args.no_threshold
    
    output_folder = os.path.join("TestMusic", "Result")
    os.makedirs(output_folder, exist_ok=True)
    
    # 출력 파일 경로 설정
    if args.output:
        output_file = args.output
    else:
        # 입력 파일명에서 확장자를 제거하고 설정 정보 추가
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        threshold_suffix = "_threshold" if use_threshold else "_original"
        segment_suffix = f"_{int(segment_duration)}s"
        output_file = os.path.join(output_folder, 
                                 f"{base_name}_emotion_analysis{threshold_suffix}{segment_suffix}.csv")

    # 입력 파일 존재 여부 확인
    if not os.path.exists(input_file):
        print(f"오류: 입력 파일이 존재하지 않습니다: {input_file}")
        return
    
    try:
        # 학습된 모델과 스케일러 로드
        model, scaler, feature_names, emotion_mapping = load_trained_model_and_scaler(args.model_dir)
        
        # MP3 파일 처리
        results_df = process_mp3_file(input_file, model, scaler, feature_names, emotion_mapping,
                                     segment_duration, use_threshold)
        
        # 예측 결과 평활화
        results_df = smooth_predictions(results_df, emotion_mapping, window_size=3)
        
        # 결과 저장
        save_results(results_df, output_file, use_threshold)
        
        print(f"\n🎉 감정 분석이 완료되었습니다!")
        print(f"📁 입력 파일: {input_file}")
        print(f"📁 출력 파일: {output_file}")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return

if __name__ == "__main__":
    # 명령행 인수가 없을 때도 실행
    import sys
    if len(sys.argv) == 1:
        print("🎵 === 개선된 MP3 감정 분석 프로그램 === 🎵")
        print("이 프로그램은 MP3 파일을 분석하여 감정을 분류합니다.")
        print("🔧 개선 사항:")
        print("  • 2초 세그먼트로 빠른 변화 감지")
        print("  • 확률 임계값 적용으로 Amusing 과도 분류 방지")
        print("  • 더 상세한 분석 결과 제공")
        print("\n📋 사용법:")
        print("1. 대화형 실행: python 3_ImprovedMP3Model.py")
        print("2. 명령행 실행: python 3_ImprovedMP3Model.py --input file.mp3 --segment 2")
        print("\n🚀 대화형 실행을 시작합니다...")
        print("-" * 60)
        
    main() 