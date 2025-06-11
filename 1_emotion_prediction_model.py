# 음악 특징 데이터를 이용한 2초 단위 감정 예측 모델 개발
# 데이터 위치: IntegratedData/ 폴더 내의 *_integrated.csv 파일들

import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("음악 특징 데이터를 이용한 2초 단위 감정 예측 모델 개발")
print("=" * 60)

# 1. 라이브러리 임포트 완료
print("1. 필요한 라이브러리 임포트 완료")

# 2. 데이터 로드 및 통합
print("\n2. 데이터 로드 및 통합 시작...")

# IntegratedData 폴더 내의 모든 *_integrated.csv 파일 찾기
file_pattern = "IntegratedData/*_integrated.csv"
csv_files = glob.glob(file_pattern)

print(f"발견된 CSV 파일 수: {len(csv_files)}개")

# 모든 CSV 파일을 하나의 데이터프레임으로 통합
dataframes = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dataframes.append(df)
        #print(f"로드 완료: {file} - 행 수: {len(df)}")
    except Exception as e:
        print(f"로드 실패: {file} - 오류: {e}")

# 모든 데이터프레임 합치기
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
    print(f"\n데이터 통합 완료!")
    print(f"전체 데이터 크기: {data.shape}")
    print(f"컬럼 수: {data.shape[1]}")
else:
    raise ValueError("CSV 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")

# 3. 데이터 전처리
print("\n3. 데이터 전처리 시작...")

# segment_id에서 song_id 추출
data['song_id'] = data['segment_id'].str.split('_').str[0]
print(f"고유한 곡 수: {data['song_id'].nunique()}개")

# 특징(X)과 타겟(y) 분리
# emotion 컬럼이 타겟, segment_id, song_id, start_time, end_time, emotion_id는 제외
feature_columns = [col for col in data.columns if col not in ['segment_id', 'song_id', 'emotion', 'emotion_id', 'start_time', 'end_time']]
X = data[feature_columns]
y = data['emotion']

print(f"특징 컬럼 수: {len(feature_columns)}")
print(f"데이터 샘플 수: {len(X)}")

# 감정 레이블 분포 확인
print(f"\n감정 레이블 분포:")
emotion_counts = y.value_counts()
print(emotion_counts)
print(f"\n감정 레이블 비율:")
print(y.value_counts(normalize=True) * 100)

# 문자열 형태의 타겟을 숫자로 변환
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 레이블 매핑 정보 저장
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"\n감정 레이블 매핑:")
for emotion, code in label_mapping.items():
    print(f"  {emotion}: {code}")

# 4. 데이터 분할 (GroupShuffleSplit 사용)
print("\n4. 데이터 분할 (GroupShuffleSplit 사용)...")

# 곡 ID를 기준으로 데이터 분할 (데이터 누수 방지)
groups = data['song_id']
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(X, y_encoded, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"훈련 데이터 크기: {X_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")

# 분할된 데이터의 곡 ID 겹침 확인
train_songs = set(data.iloc[train_idx]['song_id'])
test_songs = set(data.iloc[test_idx]['song_id'])
overlap = train_songs.intersection(test_songs)
print(f"훈련/테스트 데이터 간 곡 ID 겹침: {len(overlap)}개 (0이어야 함)")

# 분할 후 감정 분포 확인
print(f"\n훈련 데이터 감정 분포:")
train_emotions = pd.Series(y_train).map({v: k for k, v in label_mapping.items()})
print(train_emotions.value_counts())

print(f"\n테스트 데이터 감정 분포:")
test_emotions = pd.Series(y_test).map({v: k for k, v in label_mapping.items()})
print(test_emotions.value_counts())

# 5. 클래스 가중치 계산
print("\n5. 클래스 가중치 계산...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"클래스 가중치: {class_weight_dict}")

# 6. 모델 구축 (Class Weight 적용)
print("\n6. 모델 구축...")

# XGBoost 모델 생성 (class_weight 적용)
model = XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("모델 구성:")
print("  XGBClassifier with Class Weight")

# 7. 모델 훈련
print("\n7. 모델 훈련 시작...")

# 클래스 가중치를 sample_weight로 변환
sample_weights = np.array([class_weight_dict[y] for y in y_train])

# 모델 훈련 (sample_weight 적용)
model.fit(X_train, y_train, sample_weight=sample_weights)
print("모델 훈련 완료!")

# 8. 모델 예측 및 평가
print("\n8. 모델 예측 및 평가...")

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.4f}")

# Classification Report 출력
print(f"\n=== Classification Report ===")
target_names = [label_encoder.inverse_transform([i])[0] for i in sorted(np.unique(y_test))]
report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
print(report)

# Confusion Matrix 계산 및 시각화
print(f"\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Confusion Matrix 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - 음악 감정 예측 모델', fontsize=14, fontweight='bold')
plt.xlabel('예측된 감정', fontsize=12)
plt.ylabel('실제 감정', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 클래스별 성능 요약
print(f"\n=== 클래스별 성능 요약 ===")
from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=None)

performance_df = pd.DataFrame({
    '감정': target_names,
    'Precision': precision,
    'Recall': recall,
    'F1-score': fscore,
    'Support': support
})

performance_df = performance_df.round(4)
print(performance_df.to_string(index=False))

# 전체 성능 요약
print(f"\n=== 전체 성능 요약 ===")
macro_avg = {
    'Precision': precision.mean(),
    'Recall': recall.mean(),
    'F1-score': fscore.mean()
}

weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"Macro Average - Precision: {macro_avg['Precision']:.4f}, Recall: {macro_avg['Recall']:.4f}, F1-score: {macro_avg['F1-score']:.4f}")
print(f"Weighted Average - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1-score: {weighted_fscore:.4f}")
print(f"Overall Accuracy: {accuracy:.4f}")

# 클래스 가중치 적용 효과 정보 출력
print(f"\n=== 클래스 가중치 적용 효과 ===")
print("원본 훈련 데이터 클래스 분포:")
original_dist = pd.Series(y_train).value_counts().sort_index()
for i, count in enumerate(original_dist):
    emotion_name = label_encoder.inverse_transform([i])[0]
    weight = class_weight_dict[i]
    print(f"  {emotion_name}: {count}개 ({count/len(y_train)*100:.1f}%) - 가중치: {weight:.3f}")

print("\n참고: 클래스 가중치는 모델 훈련 시 적용되어 불균형 데이터 문제를 완화합니다.")

print(f"\n" + "=" * 60)
print("모델 개발 및 평가 완료!")
print("=" * 60) 