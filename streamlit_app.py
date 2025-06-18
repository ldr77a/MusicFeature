import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import librosa
import joblib
import warnings
from scipy.stats import skew, kurtosis
import tempfile
import time
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit.components.v1 as components
import base64

# Librosa 경고 메시지 필터링
warnings.filterwarnings("ignore", message="n_fft=.*is too large for input signal of length.*")

# 페이지 설정
st.set_page_config(
    page_title="음악 감정 분석기",
    page_icon="🎵",
    layout="wide",
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .segment-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #007bff;
    }
    .confidence-high { border-left-color: #28a745; }
    .confidence-medium { border-left-color: #ffc107; }
    .confidence-low { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_emotion_images():
    """감정 이미지를 로드합니다."""
    emotion_images = {}
    emotion_files = {
        'Sad': 'Sad_Emotion.png',
        'Annoying': 'Annoying_Emotion.png', 
        'Calme': 'Clame_Emotion.png',
        'Amusing': 'Amusing_Emotion.png'
    }
    
    for emotion, filename in emotion_files.items():
        img_path = os.path.join('Emotion_png', filename)
        if os.path.exists(img_path):
            emotion_images[emotion] = Image.open(img_path)
    
    return emotion_images

def image_to_base64(image):
    """PIL 이미지를 base64로 변환"""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_rhythm_game_html(results_df, emotion_images, audio_file_bytes):
    """리듬게임 HTML 생성 (타이밍 개선 최종 버전)"""
    
    audio_b64 = base64.b64encode(audio_file_bytes).decode('utf-8')
    emotion_images_b64 = {emotion: image_to_base64(img) for emotion, img in emotion_images.items()}
    results_json = results_df.to_dict('records')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            /* CSS는 이전과 동일하게 유지 */
            .game-container {{ position: relative; width: 100%; height: 500px; background: #f0f2f6; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
            .audio-player {{ position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); width: 90%; z-index: 10; }}
            .audio-player audio {{ width: 100%; }}
            .floating-note {{ position: absolute; width: 80px; height: 80px; animation: noteAnimation 1.5s ease-out forwards; pointer-events: none; z-index: 50; }}
            .floating-note img {{ width: 100%; height: 100%; border-radius: 50%; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }}
            @keyframes noteAnimation {{
                0% {{ opacity: 0; transform: scale(0.5); }}
                20% {{ opacity: 1; transform: scale(1.1); }}
                80% {{ opacity: 1; transform: scale(0.9); }}
                100% {{ opacity: 0; transform: scale(0.5); }}
            }}
        </style>
    </head>
    <body>
        <div class="game-container" id="gameContainer">
            <div class="audio-player">
                <audio id="audioPlayer" controls src="data:audio/mp3;base64,{audio_b64}"></audio>
            </div>
        </div>
        
        <script>
            const notes = {json.dumps(results_json)};
            const emotionImages = {json.dumps(emotion_images_b64)};
            
            const gameContainer = document.getElementById('gameContainer');
            const audioPlayer = document.getElementById('audioPlayer');

            // ==========================================================
            // ===            최적화된 노트 타이밍 시스템              ===
            // ==========================================================

            // 1. 더 빠른 반응을 위한 짧은 Lookahead (0.05초 = 50ms)
            const lookahead = 0.05; 
            
            // 처리해야 할 다음 노트의 인덱스
            let nextNoteIndex = 0;

            // 노트 생성 함수 (즉시 생성으로 개선)
            function spawnEmotionNote(emotion) {{
                if (!emotionImages[emotion]) return;
                const note = document.createElement('div');
                note.className = 'floating-note';
                note.style.left = `${{Math.random() * (gameContainer.offsetWidth - 90) + 5}}px`;
                note.style.top = `${{Math.random() * (gameContainer.offsetHeight - 150) + 5}}px`;
                note.innerHTML = `<img src="${{emotionImages[emotion]}}" alt="${{emotion}}">`;
                gameContainer.appendChild(note);
                setTimeout(() => note.remove(), 1500);
            }}

            // 2. 고주파 타이밍 체크 (더 정확한 동기화)
            function gameLoop() {{
                if (!audioPlayer.paused) {{
                    const currentTime = audioPlayer.currentTime;

                    // 더 적극적인 노트 생성 (lookahead 범위 내 모든 노트 즉시 처리)
                    while (nextNoteIndex < notes.length && notes[nextNoteIndex].beat_time <= currentTime + lookahead) {{
                        
                        const note = notes[nextNoteIndex];
                        const timeDiff = note.beat_time - currentTime;
                        
                        // 시간 차이가 매우 작으면 즉시 생성, 아니면 정확한 타이밍에 예약
                        if (timeDiff <= 0.02) {{  // 20ms 이내면 즉시 생성
                            spawnEmotionNote(note.emotion);
                            console.log(`⚡ ${{note.beat_time.toFixed(2)}}s: ${{note.emotion}} 노트 즉시 생성!`);
                        }} else {{
                            const delay = timeDiff * 1000;
                            setTimeout(() => {{
                                spawnEmotionNote(note.emotion);
                                console.log(`🎵 ${{note.beat_time.toFixed(2)}}s: ${{note.emotion}} 노트 생성! (지연: ${{delay.toFixed(0)}}ms)`);
                            }}, delay);
                        }}
                        
                        nextNoteIndex++;
                    }}
                }}
                
                // 더 높은 주파수로 체크 (60fps 대신 120fps 수준)
                setTimeout(() => requestAnimationFrame(gameLoop), 8);
            }}

            // 시간 이동 시 인덱스 재설정 (개선된 버전)
            audioPlayer.addEventListener('seeked', () => {{
                const currentTime = audioPlayer.currentTime;
                nextNoteIndex = 0;
                
                // 현재 시간 이후의 첫 번째 노트 찾기
                for (let i = 0; i < notes.length; i++) {{
                    if (notes[i].beat_time >= currentTime - 0.1) {{  // 0.1초 여유
                        nextNoteIndex = i;
                        break;
                    }}
                }}
                console.log(`⏭️ 시간 이동 (${{currentTime.toFixed(2)}}s). 다음 노트 인덱스: ${{nextNoteIndex}}`);
            }});

            // ==========================================================
            // ===              최적화된 타이밍 시스템 끝              ===
            // ==========================================================

            console.log('리듬게임 초기화 완료. 노트 개수:', notes.length);
            
            // 게임 루프 시작
            gameLoop();
        </script>
    </body>
    </html>
    """
    
    return html_content

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

def detect_rhythmic_onsets_balanced(file_path, target_note_count=100, chunk_duration=3.0, min_interval=0.3):
    """
    🎮 리듬게임 최적화: HPSS + 시간 균등 분배 + 간격 조절
    
    1. HPSS로 깨끗한 드럼/타악기 파트만 추출
    2. 곡을 일정 구간으로 나누어 각 구간에서 균등하게 노트 선택
    3. 최소 간격 필터링으로 연달아 나오는 노트 방지
    4. 리듬게임다운 자연스러운 노트 분포 달성
    """
    try:
        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(y=y, sr=sr)

        # 1. HPSS로 리듬 파트만 분리 (핵심 개선)
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # 2. 리듬 파트에서만 Onset Strength 계산
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        
        # 3. 모든 리듬 Onset 감지
        all_onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sr, 
            units='frames', 
            backtrack=True
        )
        
        if len(all_onset_frames) == 0:
            st.warning("리듬 온셋을 찾을 수 없습니다.")
            return []

        # 4. 시간 구간별 균등 분배 시스템
        total_chunks = int(np.ceil(duration / chunk_duration))
        notes_per_chunk = max(1, target_note_count // total_chunks)
        
        st.info(f"🎵 곡 길이: {duration:.1f}초 → {total_chunks}개 구간으로 분할")
        st.info(f"🎯 구간당 목표 노트: {notes_per_chunk}개 (총 목표: {target_note_count}개)")

        final_onset_frames = []
        actual_notes_selected = 0
        
        # 5. 각 시간 구간에서 독립적으로 최강 노트 선택
        for chunk_idx in range(total_chunks):
            start_time = chunk_idx * chunk_duration
            end_time = min((chunk_idx + 1) * chunk_duration, duration)
            
            # 현재 구간의 프레임 범위
            start_frame = librosa.time_to_frames(start_time, sr=sr)
            end_frame = librosa.time_to_frames(end_time, sr=sr)
            
            # 이 구간에 속하는 온셋들만 필터링
            chunk_onsets = []
            for frame in all_onset_frames:
                if start_frame <= frame < end_frame:
                    chunk_onsets.append({
                        'frame': frame,
                        'strength': onset_env[frame],
                        'time': librosa.frames_to_time(frame, sr=sr)
                    })
            
            # 구간 내 온셋이 있다면 강도순 정렬 후 상위 선택
            if chunk_onsets:
                # 강도 기준 내림차순 정렬
                chunk_onsets.sort(key=lambda x: x['strength'], reverse=True)
                
                # 이 구간에서 선택할 노트 수 결정
                available_notes = len(chunk_onsets)
                select_count = min(notes_per_chunk, available_notes)
                
                # 마지막 구간에서는 남은 노트 할당량 모두 사용
                if chunk_idx == total_chunks - 1:
                    remaining_quota = target_note_count - actual_notes_selected
                    select_count = min(remaining_quota, available_notes)
                
                # 선택된 온셋들의 프레임 추가
                selected_onsets = chunk_onsets[:select_count]
                for onset in selected_onsets:
                    final_onset_frames.append(onset['frame'])
                
                actual_notes_selected += select_count
                
                # 디버깅 정보
                chunk_time_range = f"{start_time:.1f}~{end_time:.1f}초"
                #st.write(f"📍 구간 {chunk_idx+1} ({chunk_time_range}): {available_notes}개 중 {select_count}개 선택")

                 # 6. 시간순 정렬
        final_onset_frames.sort()
        
        # 7. 프레임을 시간으로 변환
        onset_times_raw = librosa.frames_to_time(np.array(final_onset_frames), sr=sr)
        
        # 8. 최소 간격 필터링 (연달아 나오는 노트 방지)
        filtered_times = []
        last_time = -999  # 충분히 작은 값으로 초기화
        
        for time in onset_times_raw:
            if time - last_time >= min_interval:
                filtered_times.append(time)
                last_time = time
        
        # 9. 최종 결과 보고
        removed_count = len(onset_times_raw) - len(filtered_times)
        st.success(f" 리듬게임 노트 생성 완료!")
        st.info(f" 최종 노트: {len(filtered_times)}개 (목표: {target_note_count}개)")
        if removed_count > 0:
            st.info(f" 간격 조절: {removed_count}개 노트 제거 (최소 간격: {min_interval}초)")
        st.info(f" 평균 노트 간격: {duration/len(filtered_times):.2f}초")
        
        return filtered_times

    except Exception as e:
        st.error(f"리듬 온셋 분석 실패: {e}")
        return []

@st.cache_resource
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

def process_audio_on_beats(file_path, beat_times, model, scaler, feature_names, emotion_mapping, segment_duration=2.0):
    """
    MP3 파일을 '비트 시점'을 기준으로 처리하여 감정 분류를 수행합니다.
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
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_beats = len(beat_times)
    
    # 변경점: 고정된 시간 대신 beat_times 리스트를 순회하며 분석
    for i, beat_time in enumerate(beat_times):
        # 각 비트 시점을 중심으로 segment_duration(2초) 만큼의 분석창 설정
        start_time = max(0, beat_time - (segment_duration / 2))
        end_time = min(duration, beat_time + (segment_duration / 2))

        # 분석창의 길이가 너무 짧으면 건너뜀
        if end_time - start_time < 0.5:
            continue
            
        # 진행 상황 업데이트
        progress = (i + 1) / total_beats
        progress_bar.progress(progress)
        status_text.text(f'감정 분석 중... {i+1}/{total_beats} 비트')

        try:
            features = extract_features(y, sr, start_time, end_time)
        except Exception as e:
            st.warning(f"특징 추출 실패 (비트: {beat_time:.2f}s): {e}")
            continue
        
        selected_features = [features.get(name, 0) for name in feature_names]
        X = np.array(selected_features).reshape(1, -1)
        
        try:
            X_scaled = scaler.transform(X)
            emotion_prob = model.predict_proba(X_scaled)[0]
            
            emotion_id, confidence, _ = apply_emotion_threshold(emotion_prob, emotion_thresholds)
            emotion_name = emotion_mapping.get(emotion_id, 'Unknown')
            
        except Exception as e:
            st.warning(f"감정 예측 실패 (비트: {beat_time:.2f}s): {e}")
            continue
        
        result = {
            'beat_id': i + 1,
            'beat_time': beat_time,  # 중요: 노트가 나타날 정확한 시간
            'emotion': emotion_name,
            'confidence': confidence,
            'analysis_start': start_time, # 참고용: 분석에 사용된 오디오 시작
            'analysis_end': end_time,   # 참고용: 분석에 사용된 오디오 끝
        }
        
        # 각 감정의 확률 추가
        for idx, class_label in enumerate(model.classes_):
            prob_emotion_name = emotion_mapping.get(class_label, f'Unknown_{class_label}')
            result[f'prob_{prob_emotion_name}'] = emotion_prob[idx]
        
        results.append(result)
        
    # 진행 상황 완료
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(results)
    return df

def main():
    st.markdown('<h1 class="main-header">🎵 음악 감정 분석기</h1>', unsafe_allow_html=True)
    
    # 리듬게임 최적화 설정
    with st.sidebar:
        st.header("🎮 리듬게임 설정")
        
        target_notes = st.slider(
            "총 노트 개수",
            min_value=50,
            max_value=400,
            value=120,
            step=10,
            help="전체 곡에서 생성할 총 노트 개수. 리듬게임의 전체적인 난이도를 결정합니다."
        )
        
        chunk_duration = st.slider(
            "분배 구간 길이 (초)",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="곡을 이 길이로 나누어 각 구간에서 균등하게 노트를 선택합니다. 짧을수록 더 균등한 분배가 됩니다."
        )
        
        min_interval = st.slider(
            "최소 노트 간격 (초)",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.1,
            help="연달아 나오는 노트 사이의 최소 간격입니다. 클수록 노트가 더 여유롭게 배치됩니다."
        )
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "MP3 파일을 선택하세요",
        type=['mp3'],
        help="최대 50MB까지 업로드 가능합니다."
    )
    
    if uploaded_file is not None:
        # 파일 정보 표시
        file_details = {
            "파일명": uploaded_file.name,
            "파일 크기": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }
        
        with st.expander(" 파일 정보", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**파일명:** {file_details['파일명']}")
            with col2:
                st.write(f"**크기:** {file_details['파일 크기']}")
        
        # 음악 플레이어 (분석 전 미리보기)
        st.audio(uploaded_file, format='audio/mp3')
        
        # 분석 시작 버튼
        if st.button(" 감정 분석 시작", type="primary"):
            try:
                # 파일 내용을 먼저 읽어서 저장
                audio_file_bytes = uploaded_file.read()
                
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(audio_file_bytes)
                    tmp_file_path = tmp_file.name
                
                with st.spinner("🎮 리듬게임 최적화 분석 중..."):
                    model, scaler, feature_names, emotion_mapping = load_trained_model_and_scaler()
                    
                    # 균등 분배 리듬 분석 호출
                    note_times = detect_rhythmic_onsets_balanced(
                        tmp_file_path, 
                        target_note_count=target_notes,
                        chunk_duration=chunk_duration,
                        min_interval=min_interval
                    )

                if not note_times:
                    st.error("이 음악에서 연주 노트를 감지할 수 없습니다.")
                else:
                    # st.info 메시지는 detect_onsets 함수 내부에서 처리
                    
                    # 변경점 3: 새로운 기반 분석 함수에 note_times 전달
                    results_df = process_audio_on_beats(
                        tmp_file_path, note_times, model, scaler, feature_names, emotion_mapping
                    )
                    
                    if not results_df.empty:
                        # display_results에는 이제 strong_beats를 넘길 필요가 없습니다.
                        display_results(results_df, uploaded_file.name, audio_file_bytes)
                    else:
                        st.error("분석 결과가 없습니다. 파일을 확인해주세요.")
                
                # 임시 파일 삭제
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {str(e)}")

def display_results(results_df, filename, audio_file_bytes=None):
    """분석 결과를 표시합니다."""
    # 이제 results_df에 모든 정보가 있으므로, 총 분석 세그먼트 대신 총 노트 수를 표시
    st.success(f" 분석이 완료되었습니다! 총 {len(results_df)}개의 감정 노트를 생성했습니다.")
    
    # 감정 이미지 로드
    emotion_images = load_emotion_images()
    
    if emotion_images and audio_file_bytes:
        st.header("🎮 리듬게임 - 감정 노트")
        
        try:
            # 변경점: strong_beats 인자 없이 호출
            rhythm_game_html = create_rhythm_game_html(results_df, emotion_images, audio_file_bytes)
            components.html(rhythm_game_html, height=600)
        except Exception as e:
            st.error(f"리듬게임 로드 실패: {e}")
            st.info("일반 음악 플레이어를 대신 표시합니다.")
            st.audio(audio_file_bytes, format='audio/mp3')
    else:
        st.header("🎵 음악 플레이어")
        if audio_file_bytes:
            st.audio(audio_file_bytes, format='audio/mp3')
        else:
            st.warning("오디오 파일을 로드할 수 없습니다.")
    
if __name__ == "__main__":
    main() 