import re

# 파일 읽기
with open('pages/5_이탈_예측.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 전체 코드 길이 확인
lines = content.split('\n')
print(f"파일 전체 라인 수: {len(lines)}")

# 랜덤 데이터 생성 섹션 찾기
random_data_section = re.search(r'if input_method == "랜덤 데이터 생성":(.*?)# 푸터', content, re.DOTALL)
if random_data_section:
    section = random_data_section.group(1)
    print("랜덤 데이터 생성 섹션 찾음!")
    
    # tenure 변수 찾기
    tenure_match = re.search(r'tenure\s*=\s*st\.slider\(.*?\)', section)
    if tenure_match:
        print(f"tenure 변수 찾음: {tenure_match.group(0)}")
    else:
        print("tenure 변수를 찾을 수 없음!")
        
    # total_charges 계산 부분 찾기
    total_charges_match = re.search(r'total_charges\s*=\s*float\(tenure\)\s*\*\s*monthly_charges', section)
    if total_charges_match:
        print(f"총 요금 계산 부분 찾음: {total_charges_match.group(0)}")
        
    # 들여쓰기 문제 패턴 확인
    indentation_issues = re.findall(r'^\s{8,}(gender|tenure|contract|internet_options)', section, re.MULTILINE)
    if indentation_issues:
        print(f"들여쓰기 문제가 있는 라인: {indentation_issues}")
else:
    print("랜덤 데이터 생성 섹션을 찾을 수 없음!") 