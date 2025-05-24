# 코스탈라니 달걀모형 자동 분석 시스템 실행 방법

## 1. 파일 구조
```
프로젝트 루트/
│
├── src/
│   └── kostalany_integrated.py  # 메인 분석 파이썬 파일
│   └── (기타 관련 소스)
│
├── output/
│   └── kostalany/
│       └── kostalany_model_us_YYYYMMDD_HHMMSS.png  # 분석 결과 이미지
│       └── kostalany_model_us_YYYYMMDD_HHMMSS.md   # 분석 결과 요약(Markdown)
│
└── kostalany.sh  # 실행 스크립트
```

## 2. 실행 방법
1. 터미널에서 프로젝트 루트 디렉토리로 이동
2. 아래 명령어로 실행
   ```bash
   bash kostalany.sh
   ```
   또는 실행 권한이 있을 경우
   ```bash
   ./kostalany.sh
   ```

## 3. 동작 설명
- `kostalany.sh`는 내부적으로 다음 명령을 실행합니다:
  ```bash
  python src/kostalany_integrated.py --country=us
  ```
- 미국(US) 거시경제 지표를 기반으로 코스탈라니 달걀모형 분석을 자동 수행합니다.
- 분석 결과는 `output/kostalany/` 폴더에 이미지(.png)와 요약(.md) 파일로 저장됩니다.

## 4. 결과 확인
- `output/kostalany/` 폴더에서 최신 분석 결과 이미지와 md 파일을 확인할 수 있습니다.

## 5. 옵션 변경 예시
- 국가, 기간 등 옵션을 바꾸고 싶다면 `kostalany.sh`의 python 실행 부분을 수정하세요.
  - 예시(한국, 3년치):
    ```bash
    python src/kostalany_integrated.py --country=korea --history --history_years=3
    ```

---

**요약:**
- `kostalany.sh`를 실행하면 미국 거시경제 지표 기반 코스탈라니 달걀모형 분석이 자동으로 수행되고,
- 결과가 output/kostalany 폴더에 이미지와 md 파일로 저장됩니다.

추가 설명이나 옵션별 사용법이 필요하시면 언제든 말씀해 주세요! 