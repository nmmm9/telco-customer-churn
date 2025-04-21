import os

# 원본 파일과 수정된 파일의 경로
original_file = 'pages/5_이탈_예측.py'
fixed_file = 'pages/5_이탈_예측_fixed.py'

# 파일이 존재하는지 확인
if not os.path.exists(original_file):
    print(f"원본 파일을 찾을 수 없습니다: {original_file}")
    exit(1)
    
if not os.path.exists(fixed_file):
    print(f"수정된 파일을 찾을 수 없습니다: {fixed_file}")
    exit(1)
    
# 두 파일 읽기
with open(original_file, 'r', encoding='utf-8') as f1:
    original_lines = f1.readlines()
    
with open(fixed_file, 'r', encoding='utf-8') as f2:
    fixed_lines = f2.readlines()
    
# 파일 길이 비교
print(f"원본 파일: {len(original_lines)}줄")
print(f"수정된 파일: {len(fixed_lines)}줄")

# 랜덤 데이터 생성 섹션 찾기
def find_section(lines, start_pattern, end_pattern):
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if start_pattern in line and start_idx == -1:
            start_idx = i
        elif end_pattern in line and start_idx != -1:
            end_idx = i
            break
            
    return start_idx, end_idx

# 랜덤 데이터 생성 섹션 범위 찾기
original_start, original_end = find_section(original_lines, 'if input_method == "랜덤 데이터 생성":', '# 푸터')
fixed_start, fixed_end = find_section(fixed_lines, 'if input_method == "랜덤 데이터 생성":', '# 푸터')

print(f"원본 파일의 랜덤 데이터 생성 섹션: {original_start}-{original_end}줄")
print(f"수정된 파일의 랜덤 데이터 생성 섹션: {fixed_start}-{fixed_end}줄")

# 특정 부분 출력 (기본 정보 탭)
def print_section(lines, title, start_pattern, range_after=15):
    for i, line in enumerate(lines):
        if start_pattern in line:
            print(f"\n{title} (줄 {i}부터):")
            for j in range(i, min(i + range_after, len(lines))):
                print(f"{j+1:4d}: {lines[j].rstrip()}")
            return
    print(f"{title}: 패턴 '{start_pattern}'을 찾을 수 없습니다.")

# 기본 정보 탭 출력
print_section(original_lines, "원본 파일 - 기본 정보 탭", "# 기본 정보 탭")
print_section(fixed_lines, "수정된 파일 - 기본 정보 탭", "# 기본 정보 탭")

# 서비스 정보 탭 출력
print_section(original_lines, "원본 파일 - 서비스 정보 탭", "# 서비스 정보 탭")
print_section(fixed_lines, "수정된 파일 - 서비스 정보 탭", "# 서비스 정보 탭")

# 결제 정보 탭 출력
print_section(original_lines, "원본 파일 - 결제 정보 탭", "# 결제 정보 탭")
print_section(fixed_lines, "수정된 파일 - 결제 정보 탭", "# 결제 정보 탭")

# 파일 검증
print("\n파일 검증:")

# 수정된 파일에서 indentation 에러가 있는지 확인 (간단한 휴리스틱)
indentation_issues_fixed = []
for i, line in enumerate(fixed_lines[fixed_start:fixed_end]):
    line_num = i + fixed_start + 1
    if line.strip().startswith('gender =') and not line.startswith('            gender ='):
        indentation_issues_fixed.append(f"줄 {line_num}: gender = 들여쓰기 문제")
    if line.strip().startswith('tenure =') and not line.startswith('            tenure ='):
        indentation_issues_fixed.append(f"줄 {line_num}: tenure = 들여쓰기 문제")
    if line.strip().startswith('contract =') and not line.startswith('            contract ='):
        indentation_issues_fixed.append(f"줄 {line_num}: contract = 들여쓰기 문제")

if indentation_issues_fixed:
    print("수정된 파일에 들여쓰기 문제가 있을 수 있습니다:")
    for issue in indentation_issues_fixed:
        print(f"  - {issue}")
else:
    print("수정된 파일에서 주요 들여쓰기 문제가 발견되지 않았습니다.")

# 파일 교체 안내
print("\n파일을 교체하려면 다음 명령을 실행하세요:")
print("import shutil")
print("shutil.copy('pages/5_이탈_예측_fixed.py', 'pages/5_이탈_예측.py')")
print("print('파일이 성공적으로 교체되었습니다.')") 