import pickle
import os
import json

print("현재 디렉토리:", os.getcwd())

try:
    # 파일 로드
    with open('models/moonyoung_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    print(f"전처리기 로드 성공!")
    print(f"전처리기 타입: {type(preprocessor)}")
    
    # 전처리기가 딕셔너리라면 키 확인
    if isinstance(preprocessor, dict):
        print("전처리기 키:", list(preprocessor.keys()))
        
        # 스케일러 확인
        if 'scaler' in preprocessor:
            print(f"스케일러 타입: {type(preprocessor['scaler'])}")
        
        # 특성 이름 확인
        if 'feature_names' in preprocessor:
            feature_names = preprocessor['feature_names']
            print(f"특성 수: {len(feature_names)}")
            print(f"특성 이름 목록: {feature_names}")
        else:
            print("feature_names 키가 없습니다.")
            
        # 특성 수 확인
        for key, value in preprocessor.items():
            try:
                if hasattr(value, "feature_names_in_"):
                    print(f"{key}의 특성 이름: {value.feature_names_in_}")
                    print(f"{key}의 특성 수: {len(value.feature_names_in_)}")
            except Exception as e:
                print(f"{key} 특성 정보 확인 실패: {e}")
    else:
        # scikit-learn 전처리기의 경우
        try:
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
                print(f"특성 수: {len(feature_names)}")
                print(f"특성 이름 목록: {list(feature_names)}")
            elif hasattr(preprocessor, "feature_names_in_"):
                feature_names = preprocessor.feature_names_in_
                print(f"입력 특성 수: {len(feature_names)}")
                print(f"입력 특성 이름 목록: {list(feature_names)}")
        except Exception as e:
            print(f"특성 정보 확인 실패: {e}")
    
    # 전처리기 내부 구조 요약해서 파일로 저장
    summary = {
        "type": str(type(preprocessor)),
        "structure": str(preprocessor)[:1000] + "..." if len(str(preprocessor)) > 1000 else str(preprocessor)
    }
    
    with open('preprocessor_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("전처리기 정보가 preprocessor_summary.json 파일에 저장되었습니다.")
    
except Exception as e:
    print(f"오류 발생: {e}") 