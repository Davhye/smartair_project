# SmartAir: 실시간 공기질 예측 및 API 제공 플랫폼

## 📌 프로젝트 개요

이 프로젝트는 공기질 데이터( PM10, CO2, TVOC)를 기반으로 실시간 예측을 수행하고, 예측 결과를 FastAPI 서버를 통해 전달하는 시스템입니다.  
모델 학습은 Google Colab에서 진행되었으며, 예측 API는 Python 기반 FastAPI로 구현하였습니다.

---

## 🧠 모델 개발 (Google Colab)

- TensorFlow/Keras 기반의 딥러닝 예측 모델
- 입력 데이터: S3에 저장된 실시간 공기질 JSON 데이터
- 정규화 처리, 시계열 구성 후 1시간 단위 예측
- 모델 성능 검증: MAE, RMSE, R² 등

[📎 Colab 파일 보기]([https://colab.research.google.com/drive/your_colab_link_here](https://colab.research.google.com/drive/1Xn7auVeNUPyOFPFwBX6cxtmE7nmH6w00?usp=sharing))

---

## 🌐 API 서버 구성 (FastAPI + Python)

- FastAPI 서버로 예측 결과 실시간 전송
- Docker로 컨테이너화 후 GCP 인스턴스에 배포
- S3에서 예측 입력 데이터 로딩 및 전처리
- JSON 형식으로 예측값 반환

---

## ⚙️ 실행 구조
📁 app/
├── avgpredict/avgoutlier.py # FastAPI 서버 메인
├── models/
│ └── pm10_model.h5
│ └── co2_model.h5
│ └── tvoc_model.h5
Dockerfile
requirements.txt
.env
README.md

---

## 📦 배포
Docker로 패키징 후 GCP VM에 배포
예측 결과를 외부 클라이언트에 실시간 POST


### 📈 향후 개선 방향
지역별/센서별 또는 장소별 예측 정밀도 향상
이상감지 시스템 정밀화
