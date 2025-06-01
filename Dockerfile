# 베이스 이미지
FROM python:3.10

# 작업 디렉터리 설정
WORKDIR /app

# 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드와 모델 파일 복사
COPY avgoutlier.py .
COPY pm10_model.h5 .
COPY co2_model.h5 .
COPY tvoc_model.h5 .

CMD ["python", "avgoutlier.py"]