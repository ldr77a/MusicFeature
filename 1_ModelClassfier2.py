import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import time

# 한글 폰트 설정 함수
def set_korean_font():
    """
    matplotlib에서 한글 폰트를 사용할 수 있도록 설정
    """
    # 시스템 폰트 경로
    font_dirs = [os.path.join(os.environ['WINDIR'], 'Fonts')]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    
    # 나눔고딕 또는 맑은 고딕 폰트 찾기
    for font_file in font_files:
        if 'malgun' in font_file.lower():  # 맑은 고딕
            font_path = font_file
            break
        elif 'nanumgothic' in font_file.lower():  # 나눔고딕
            font_path = font_file
            break
    else:
        # 적절한 폰트를 찾지 못했을 때 기본 설정 사용
        print("경고: 한글 폰트를 찾을 수 없습니다. 그래프에 한글이 제대로 표시되지 않을 수 있습니다.")
        return
    
    # 폰트 등록
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    
    # 음수 표시 문제 해결
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"한글 폰트 '{font_name}'가 설정되었습니다.")

# 프로그램 시작 시 한글 폰트 설정
set_korean_font()

def load_integrated_data(data_dir="IntegratedData"):
    """
    통합 데이터 파일 로드
    
    Args:
        data_dir (str): 통합 데이터 디렉토리 경로
        
    Returns:
        pd.DataFrame: 통합된 데이터프레임
    """
    print(f"'{data_dir}' 폴더에서 통합 데이터 로드 중...")
    
    all_data = []
    
    # 모든 통합 데이터 파일 가져오기
    integrated_files = glob.glob(os.path.join(data_dir, "*_integrated.csv"))
    
    if len(integrated_files) == 0:
        print(f"오류: '{data_dir}' 폴더에서 통합 데이터 파일을 찾을 수 없습니다.")
        return None
    
    # 각 파일 로드 및 결합
    for file in integrated_files:
        try:
            # 파일 읽기
            df = pd.read_csv(file)
            # 파일 이름에서 song_id 추출
            song_id = os.path.basename(file).split("_")[0]
            df['song_id'] = song_id
            
            # 기본 검증
            if 'emotion' not in df.columns or 'emotion_id' not in df.columns:
                print(f"경고: {file} 파일에 감정 레이블이 없습니다. 건너뜁니다.")
                continue
                
            # 데이터 추가
            all_data.append(df)
        except Exception as e:
            print(f"오류: {file} 파일 로드 중 문제 발생 - {e}")
    
    # 모든 데이터 결합
    if not all_data:
        print("오류: 처리할 데이터가 없습니다.")
        return None
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"총 {len(combined_data)}개의 데이터 포인트가 로드되었습니다.")
    
    return combined_data

def get_all_features(data):
    """
    데이터에서 모든 음악적 특징을 추출
    
    Args:
        data (pd.DataFrame): 통합 데이터
        
    Returns:
        list: 모든 특징 목록
    """
    # 메타데이터 컬럼 제외하고 모든 특징 추출
    meta_columns = ['segment_id', 'song_id', 'start_time', 'end_time', 'emotion', 'emotion_id']
    all_features = [col for col in data.columns if col not in meta_columns]
    
    print(f"총 {len(all_features)}개의 음악적 특징을 사용합니다:")
    
    # 특징들을 카테고리별로 분류하여 출력
    feature_categories = {
        'MFCC': [f for f in all_features if 'MFCC' in f],
        'Chroma': [f for f in all_features if any(note in f for note in ['C_', 'D_', 'E_', 'F_', 'G_', 'A_', 'B_'])],
        'Spectral': [f for f in all_features if 'spectral' in f or 'contrast' in f],
        'Rhythm': [f for f in all_features if any(keyword in f for keyword in ['tempo', 'onset', 'rhythm'])],
        'Energy': [f for f in all_features if any(keyword in f for keyword in ['rms', 'band_energy', 'harmonic', 'percussive'])],
        'Other': [f for f in all_features if not any(cat in f for cat in ['MFCC', 'spectral', 'contrast', 'tempo', 'onset', 'rhythm', 'rms', 'band_energy', 'harmonic', 'percussive']) 
                 and not any(note in f for note in ['C_', 'D_', 'E_', 'F_', 'G_', 'A_', 'B_'])]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"  {category}: {len(features)}개")
    
    return all_features

def train_with_smote(data, all_features, output_dir="AllFeaturesModel_SMOTE"):
    """
    SMOTE를 사용하여 클래스 불균형을 해소하고 모든 음악적 특징을 사용하여 모델 학습
    
    Args:
        data (pd.DataFrame): 원본 데이터
        all_features (list): 모든 음악적 특징 목록
        output_dir (str): 결과 저장 디렉토리
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' 폴더를 생성했습니다.")
    
    print(f"\n총 {len(all_features)}개의 모든 음악적 특징을 사용하고 SMOTE로 클래스 불균형을 해소하여 모델 학습 중...")
    
    # 특징과 레이블 분리
    X_selected = data[all_features].values
    y = data['emotion_id'].values
    
    # 클래스 분포 확인
    unique, counts = np.unique(y, return_counts=True)
    print("\n원본 데이터의 클래스 분포:")
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"클래스 {label} (감정: {data['emotion'].iloc[np.where(data['emotion_id'] == label)[0][0]]}) : {count}개")
    
    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # 훈련/테스트 세트 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"훈련 세트 크기: {X_train.shape}, 테스트 세트 크기: {X_test.shape}")
    
    # SMOTE를 사용하여 클래스 불균형 해소 (훈련 세트에만 적용)
    print("\nSMOTE를 사용하여 클래스 불균형 해소 중...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # SMOTE 적용 후 클래스 분포 확인
    unique_after, counts_after = np.unique(y_train_resampled, return_counts=True)
    print("\nSMOTE 적용 후 훈련 데이터의 클래스 분포:")
    emotion_names = {emotion_id: emotion_name for emotion_id, emotion_name in 
                    zip(data['emotion_id'].unique(), data.loc[data['emotion_id'].isin(data['emotion_id'].unique()), 'emotion'].unique())}
    
    for label, count in zip(unique_after, counts_after):
        print(f"클래스 {label} (감정: {emotion_names.get(label, '알 수 없음')}) : {count}개")
    
    print(f"\n원본 훈련 데이터 크기: {X_train.shape} -> SMOTE 적용 후 훈련 데이터 크기: {X_train_resampled.shape}")
    
    # Random Forest 모델 학습
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)
    
    train_time = time.time() - start_time
    print(f"모델 학습 완료. 소요 시간: {train_time:.2f}초")
    
    # 모델 성능 평가 (원본 테스트 세트로 평가)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"모델 정확도: {accuracy:.4f}")
    
    # 분류 보고서
    emotion_labels = sorted(data['emotion'].unique())
    target_names = [str(label) for label in emotion_labels]
    
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\n분류 보고서:")
    print(report)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('혼동 행렬 (SMOTE 적용)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smote_confusion_matrix.png'), dpi=300)
    
    # 모든 특징 중요도 시각화 (상위 30개만 표시)
    feature_importances = model.feature_importances_
    all_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # 상위 30개만 시각화 (너무 많으면 보기 어려우므로)
    top30_for_plot = all_importance_df.head(30)
    
    plt.figure(figsize=(12, 12))
    sns.barplot(x='Importance', y='Feature', data=top30_for_plot)
    plt.title('상위 30개 특징 중요도 (SMOTE 적용 - 전체 특징 사용)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smote_feature_importance.png'), dpi=300)
    
    # 전체 특징 중요도를 CSV로 저장
    all_importance_df.to_csv(os.path.join(output_dir, 'all_feature_importance.csv'), index=False)
    print(f"전체 {len(all_features)}개 특징의 중요도가 CSV 파일로 저장되었습니다.")
    
    # 모델 저장
    joblib.dump(model, os.path.join(output_dir, 'smote_model2.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'smote_scaler2.pkl'))
    
    # 특징 목록 저장
    with open(os.path.join(output_dir, 'smote_features.txt'), 'w') as f:
        for feature in all_features:
            f.write(f"{feature}\n")
    
    # 텍스트 보고서 저장
    with open(os.path.join(output_dir, 'smote_model_report.txt'), 'w') as f:
        f.write(f"SMOTE를 사용한 모델 보고서\n")
        f.write(f"=========================\n\n")
        f.write(f"모델 정확도: {accuracy:.4f}\n")
        f.write("\n분류 보고서:\n")
        f.write(report)
        f.write("\n원본 데이터의 클래스 분포:\n")
        for i, (label, count) in enumerate(zip(unique, counts)):
            f.write(f"클래스 {label} (감정: {data['emotion'].iloc[np.where(data['emotion_id'] == label)[0][0]]}) : {count}개\n")
        f.write("\nSMOTE 적용 후 훈련 데이터의 클래스 분포:\n")
        for label, count in zip(unique_after, counts_after):
            f.write(f"클래스 {label} (감정: {emotion_names.get(label, '알 수 없음')}) : {count}개\n")
        f.write("\n사용된 특징 목록:\n")
        for i, feature in enumerate(all_features, 1):
            f.write(f"{i}. {feature}\n")
    
    print(f"\nSMOTE 적용 모델이 '{output_dir}' 폴더에 저장되었습니다.")
    return model, scaler

def main():
    # 출력 디렉토리 설정
    output_dir = "Top20Model_SMOTE"
    
    # 데이터 로드
    data = load_integrated_data()
    
    if data is None:
        print("데이터 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 모든 음악적 특징 가져오기
    all_features = get_all_features(data)
    
    if all_features is None or len(all_features) == 0:
        print("사용할 수 있는 음악적 특징이 없습니다. 프로그램을 종료합니다.")
        return
    
    print(f"\n총 {len(all_features)}개의 특징을 사용하여 모델을 학습합니다.")
    
    # SMOTE를 사용하여 모든 특징으로 모델 학습
    train_with_smote(data, all_features, output_dir)
    
    print(f"\n모든 음악적 특징을 사용한 SMOTE 모델 학습이 완료되었습니다. 결과는 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 