# 외부 모델 통합 가이드

이 디렉토리는 외부에서 개발된 모델을 프로젝트에 통합하기 위한 스크립트를 포함하고 있습니다.

## 사용 가능한 외부 모델

1. **MoonyoungStacking** - 고객 이탈 예측을 위한 스태킹 앙상블 모델
   - 기반 알고리즘: RandomForest, XGBoost, LogisticRegression(메타 모델)
   - 성능: F1 점수 약 84%, ROC AUC 약 92%
   - 특징: SMOTE 기법을 활용한 클래스 불균형 해소, 파생변수 추가 적용

## 외부 모델 학습 방법

외부 모델을 학습하고 프로젝트에 통합하려면 다음 단계를 따르세요:

1. `train_moonyoung_model.py` 스크립트 실행:
   ```
   python external_models/train_moonyoung_model.py
   ```

2. 학습된 모델은 다음 파일로 저장됩니다:
   - `models/moonyoung.pkl`: 학습된 모델
   - `models/moonyoung_preprocessor.pkl`: 전처리 파이프라인
   - `models/moonyoung_meta.json`: 모델 메타데이터

3. 학습 후에는 자동으로 `models/model_results.csv` 파일에 성능 지표가 추가됩니다.

4. Streamlit 앱을 실행하면 "이탈 예측" 페이지에서 MoonyoungStacking 모델을 선택할 수 있습니다.

## 새로운 외부 모델 추가 방법

새로운 외부 모델을 추가하려면:

1. 이 디렉토리에 새로운 학습 스크립트를 생성하세요 (예: `train_mymodel.py`).
2. 모델 학습 및 저장 로직을 구현하세요.
3. 모델의 성능 메트릭을 `models/model_results.csv`에 추가하는 코드를 포함하세요.
4. 필요한 경우 `pages/5_이탈_예측.py`의 `preprocess_data_for_model` 함수에 새 모델에 대한 전처리 로직을 추가하세요.

## 참고 사항

- 외부 모델은 기존 모델과 동일한 인터페이스(fit, predict, predict_proba)를 구현해야 합니다.
- 모델 저장에는 joblib 또는 pickle을 사용하세요.
- 모델 메타데이터는 JSON 형식으로 저장하는 것이 좋습니다. 