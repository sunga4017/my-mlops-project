# 1일차: Python 문법 복습 및 Numpy 기초 학습. GitHub 계정 생성 및 Git 기본 명령어 숙달.

## 1. 학습 목표
* Python 문법 복습.
* `Numpy` 라이브러리 설치 및 `ndarray` 배열 연산 숙달.
* `Git` 설치, `GitHub` 저장소 생성 및 `push`까지의 기본 명령어 숙달.
* `.gitignore`를 이용한 불필요 파일(폴더) 관리 적용.

## 2. Python 문법
* **정보처리기 실기 교재**를 통해 복습 완료.

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

# 2일차: Pandas 기초 학습. PyTorch 환경 설정 및 Tensor 기본 조작.

## 1. 학습 목표
* `Pandas` 라이브러리 설치 및 `DataFrame` 기초 사용법 숙달.
* `Pandas`의 핵심 기능인 **불리언 인덱싱**과 **`groupby`**의 작동 원리 이해.
* `PyTorch` (GPU 버전) 설치 및 GPU 사용 가능 여부 확인.
* MLOps의 핵심 개념인 **'학습(Training)'**과 **'추론(Inference)'**의 차이 이해.
* `PyTorch`의 **`model.eval()`**과 **`torch.no_grad()`**의 차이점 및 MLOps에서의 사용 이유 이해.

## 2. Pandas 기초
* **설치:** PyCharm 터미널에서 `pip install pandas` 실행.
* **핵심:** `Pandas`는 데이터를 `DataFrame`(엑셀 시트) 형태로 다루며, 데이터 분석 및 전처리에 필수적인 도구.
* **사용법 (기본):**
    ```python
    import pandas as pd

    # 딕셔너리 -> DataFrame (엑셀 시트) 생성
    data = {'이름': ['김철수', '이영희', '박지성'], '나이': [25, 30, 28], '도시': ['서울', '부산', '서울']}
    df = pd.DataFrame(data)
        
    # 맨 위 5줄 확인
    print(df.head())
        
    # 특정 열(Column) 선택
    print(df['이름'])
    ```
* **핵심 문법 (데이터 필터링 및 분석):**
    * **조건 필터링 (Boolean Indexing):** `df[df['나이'] > 25]`
        * `df['나이'] > 25`가 `[False, True, True]` 같은 **'필터 마스크(Series)'**를 생성.
        * `df[ (필터 마스크) ]`가 `True`인 **행(Row)**만 선택함.
    * **그룹 분석 (Group By):** `df.groupby('도시')['나이'].mean()`
        * '도시' 별로 데이터를 묶고(`groupby`), 각 그룹의 '나이' 열(`['나이']`)에 대해 평균(`mean()`)을 계산.

## 3. PyTorch 기초 (환경 설정 및 Tensor)

* **GPU 환경 설정 (MLOps의 첫 단계):**
    1.  **`nvidia-smi`** 명령어로 현재 로컬 **CUDA 드라이버 버전 (11.0)** 확인.
    2.  최신 PyTorch (`CUDA 13.0`)와 버전이 맞지 않아, **NVIDIA 그래픽 드라이버 업데이트** 진행. (MLOps 환경 통일의 중요성)
    3.  재부팅 후 `nvidia-smi`로 **CUDA 버전 (13.0)**이 올바르게 업데이트된 것을 재확인.
    4.  `pytorch.org` 공식 사이트에서 로컬 환경에 맞는 옵션(`Stable`, `Pip`, `CUDA 13.0`)을 선택하여, 생성된 `pip install ...` 명령어로 PyTorch 설치 완료.

* **핵심 (`Tensor`):**
    * `Numpy`의 `ndarray`와 99% 동일 (**단일 `dtype`**만 저장 가능).
    * **차이점 1 (GPU 연산):** `.to('cuda')`로 GPU에서 초고속 병렬 연산 가능 (NumPy는 불가).
    * **차이점 2 (자동 미분):** 딥러닝 **'학습'**을 위해 **'계산 기록(history)을 추적'**할 수 있음 (`loss.backward()`의 기반).

* **사용법 (기본):**
    ```python
    import torch
    import numpy as np

    # 1. Numpy -> PyTorch Tensor (데이터는 아직 CPU에 있음)
    np_array = np.array([1, 2, 3])
    torch_tensor = torch.from_numpy(np_array)

    # 2. GPU 사용 가능 여부 확인
    is_cuda_available = torch.cuda.is_available()
    print(f"GPU 사용 가능 여부: {is_cuda_available}") # True 확인

    # 3. 텐서를 GPU로 '보내기' (수동 명령)
    if is_cuda_available:
        device = torch.device('cuda')
        tensor_on_gpu = torch_tensor.to(device)
        print(f"GPU로 보낸 텐서: {tensor_on_gpu}")
    ```

* **MLOps 추론(Inference) 표준 코드:**
    * '추론'은 이미 학습된 모델을 **'사용'**하는 과정. '학습' 시 필요했던 '계산 기록(추적)' 기능이 불필요하며, 오히려 속도를 저하 시킴.
    * **`model.eval()`:** 모델을 '추론 모드'로 전환 (Dropout 비활성화, BatchNorm 통계 고정).
    * **`torch.no_grad()`:** '계산 기록(자동 미분)'을 끄는 컨텍스트. 불필요한 메모리 사용을 막고 **속도를 향상**시킴. (추론 시 `eval()`과 `no_grad()`는 항상 세트로 사용)
    ```python
    # MLOps 추론 코드의 정석
    model = ... # (다음 주차에 배울 모델)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval() # 1. 모델을 추론 모드로
    with torch.no_grad(): # 2. 자동 미분(추적) 중지
        # (1, 3, 224, 224) 모양의 가짜 이미지 텐서를 'CPU'에 생성
        fake_image = torch.rand(1, 3, 224, 224) 
        
        # 가짜 이미지를 GPU로 보냄
        input_tensor = fake_image.to(device)
        
        # 추론 실행 (모델과 데이터가 모두 GPU에 있음)
        output = model(input_tensor)
    `````