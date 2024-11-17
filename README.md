# ctr-pred-sys
광고 회사인 Criteo 데이터셋으로 CTR Prediction을 진행하는 시스템을 구축
3가지의 다른 방식의 모델을 사용하여 CTR Prediction 진행
코드 품질과 스타일의 일관성 유지를 위해 **flake8**(코드 린팅)과 **black**(코드 포맷팅) 사용

### 프로젝트 구조
(TO-DO)


### 사전 준비

아래 항목들이 설치되어 있는지 확인하세요:

- **Python**: 3.8 이상
- **Poetry**: 최신 버전 ([설치 가이드](https://python-poetry.org/docs/))

### 셋팅 방법
1. 레포지토리 클론:
```bash
   git clone https://github.com/your-username/your-repo.git](https://github.com/f-lab-edu/ctr-pred-sys.git
   cd ctr-pred-sys
```

2. 프로젝트 의존성 설치
```
poetry install
```

3. 가상환경 활성화
```
poetry shell
```

4. flake8 실행 (설정 파일은 root directory의 .flake8 파일 참조)
```
poetry run flake8
```

5. VS Code를 활용한 black 셋팅
 - 설치
    - 좌측 Extension 클릭
    - 'Black'검색
    - 'Black Formatter' 설치
 - 설정
    - 상단 Code 클릭
    - settings 클릭
    - 'Default Formatter' 검색
    - 'Black Formatter' 기본값 설정
    - 'Format on save' 검색
    - 'Format on Save' 활성화
   
   

6. black 코드 포맷팅 적용 (특정 파일에 적용하고 싶은 경우 . 대신 파일명 명시)
```
poetry run black .
```

7. 포맷팅 확인
```
poetry run black --check .
```


