import numpy as np

# 1. 파이썬 리스트
python_list = [1, 2, 3]

# 2. 넘파이 배열 (ndarray)
numpy_array = np.array(python_list)

# 3. 넘파이로 '한 번에' 계산하기
new_array = numpy_array + 10

# 결과 출력
print("파이썬 리스트:", python_list)
print("넘파이 배열:", numpy_array)
print("넘파이 계산 결과:", new_array)