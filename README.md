# 1일차: Python 문법 복습 및 Numpy 기초 학습. GitHub 계정 생성 및 Git 기본 명령어 숙달.

## 1. 학습 목표
* Python 핵심 문법(List, Dict, def, class) 복습.
* `Numpy` 라이브러리 설치 및 `ndarray` 배열 연산 숙달.
* `Git` 설치, `GitHub` 저장소 생성 및 `push`까지의 기본 명령어 숙달.
* `.gitignore`를 이용한 불필요 파일(폴더) 관리 적용.

## 2. Python 핵심 문법
* **자료구조:** `List`, `Dictionary` (API JSON 데이터 처리 기반).
* **함수 (def):** 코드 재사용을 위한 `def` 정의.
* **클래스 (class):** `PyTorch`, `FastAPI`의 기본 구조 (`__init__`, 메서드) 파악.

## 3. Numpy 기초
* **설치:** PyCharm 터미널에서 `pip install numpy` 실행.
* **핵심:** AI/딥러닝의 대규모 숫자 연산을 위한 `ndarray`(Numpy 배열) 사용.
* **사용법:** `for`문 없는 빠른 배열 연산(벡터화).
```python
import numpy as np

my_array = np.array([1, 2, 3])
result = my_array + 10 # for문 없이 계산

# 결과: [11 12 13]
```

## 4. Git & GitHub 기초
* **Git 설치:** `git-scm.com`에서 Git 설치 및 `git --version` 확인.
* **GitHub 생성:** `GitHub.com` 회원가입 및 `my-mlops-project` 저장소 생성.
* **핵심 명령어:**
    * `git init`: 로컬 저장소 초기화.
    * `git remote add origin [주소]`: 원격 저장소 연결.
    * `git add .`: 변경 사항을 'Staging Area'에 추가.
    * `git commit -m "메시지"`: 'Commit' (로컬 저장).
    * `git push origin master`: 'Push' (원격 업로드).

## 5. .gitignore (무시 규칙)
* **역할:** `.idea/` (PyCharm 설정), `*_practice.py` (연습 파일) 등을 무시하는 규칙 정의.
* **적용:**
    1. `git rm -r --cached .idea` 명령어로 Git 기억(Cache)에서 `.idea` 폴더 삭제.
    2. `.gitignore` 파일 자체는 `add` -> `commit` -> `push` 하여 원격 저장소에 규칙 적용 완료.