from fastapi import FastAPI
from fastapi.responses import JSONResponse
import boto3
import json
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import models
from keras import layers
from tensorflow.python.keras.models import load_model
from datetime import datetime, timedelta, timezone
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import requests
from collections import defaultdict
BASE_URL = "https://smartair.site"
app = FastAPI()

# S3 접근 정보 (하드코딩 주의)
access_key_id = 'AKIAURAQUYAOMJQ7K5EV'
secret_access_key = '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'
region = 'ap-northeast-2'
bucket_name = 'smartair-bucket'

# 모델 로드
pm10_model = keras.models.load_model("pm10_model.h5", compile=False)
co2_model = keras.models.load_model('co2_model.h5', compile=False)
tvoc_model = keras.models.load_model('tvoc_model.h5', compile=False)

# 정규화 범위
pm10_min, pm10_max = 0.0, 133.0
co2_min, co2_max = 400.0, 3229.0
tvoc_min, tvoc_max = 0.0, 500.0

# 정규화 및 복원 함수
def normalize(value, min_v, max_v):
    return (value - min_v) / (max_v - min_v)

def denormalize(value, min_v, max_v):
    return float(value * (max_v - min_v) + min_v)

login_endpoint = "/login"
# 로그인 데이터
login_data = {
    "email": "admin@example.com",
    "password": "123"
}

#토근 발급 부분
try:
    # 로그인 요청 보내기
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"{BASE_URL}{login_endpoint}",
        json=login_data,
        headers=headers
    )

    # 로그인 성공 시 토큰 추출
    if response.status_code == 200:
        token = response.json().get("accessToken")
        print("✅ 토큰 발급 성공")
        print("Access Token:", token)
    else:
        print(f"🚫 로그인 실패 ({response.status_code}):")
        print(response.json())

except Exception as e:
    print(f"⚠️ 로그인 요청 실패: {e}")

getid_endpoint = "/sensorMappingWithRoom"
#센서아이디 받아오기기
try:
    # GET 요청 보내기
    headers = {'accept': '*/*',
                    'Authorization': f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}{getid_endpoint}")
    
    # 응답 상태 확인
    if response.status_code == 200:
        # 응답 내용 출력
        print("🚀 응답 내용:", response.text)
        data = response.json()
        
        # sensorSerialNumber만 추출
        serial_numbers = [sensor["sensorSerialNumber"] for sensor in data]
        print("Sensor Serial Numbers:", serial_numbers)
    else:
        print(f"🚫 서버 오류 ({response.status_code}):")
        print(response.text)

except Exception as e:
    print(f"⚠️ 요청 실패: {e}")

# 이상치 임계치 (절대값)
thresholds_abs = {
    "pm10": 14,
    "co2": 336,
    "tvoc": 140
}

s3 = boto3.client(
    's3',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name=region
)

headers = {
    'accept': '*/*',
    'Content-Type': "application/json",
    'Authorization': f"Bearer {token}"  # 토큰은 로그인 후 받아온 값
}

outlier_endpoint = "/api/reports/anomaly"
# 이상치 감지 함수 (1개 센서)
def process_sensor_anomalies_and_post(bucket_name, serial_number, s3_client, 
                                      models_dict, mins_dict, maxs_dict):
    prefix = f"airQuality/{serial_number}/"
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        files = response.get('Contents', [])
        if not files:
            print(f"⚠️ 시리얼번호 {serial_number} 경로에 파일이 없습니다.")
            return

        files.sort(key=lambda x: x['Key'].split('/')[-1], reverse=True)
        files = files[:300]  

        hourly_data = defaultdict(list)
        for file in files:
            try:
                key = file['Key']
                timestamp_str = key.split('/')[-1].replace('.json', '').split('.')[0]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)

                resp = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = resp['Body'].read().decode('utf-8')
                data = json.loads(content)

                sensor_id = data.get("id") or data.get("sensorSerialNumber", serial_number)

                pm10_avg = np.mean([
                    data.get("pt1", {}).get("pm100_standard", 0),
                    data.get("pt2", {}).get("pm100_standard", 0)
                ])
                co2 = data.get("eco2", 0)
                tvoc = data.get("tvoc", 0)

                hourly_data[hour_key].append((pm10_avg, co2, tvoc))
            except Exception as e:
                print(f"⚠️ 파일 처리 오류 ({file['Key']}): {e}")

        sorted_hours = sorted(hourly_data.keys(), reverse=True)
        sequence_hours = []
        for hour in sorted_hours:
            if len(sequence_hours) >= 5:
                break
            sequence_hours.append(hour)
            
        print("🕒 평균값 생성에 사용된 시간대:")
        for h in sequence_hours:
            print(" -", h.strftime("%Y-%m-%dT%H:%M:%S"))

        if len(sequence_hours) < 5:
            print(f"⚠️ 평균값 시퀀스 부족 (현재: {len(sequence_hours)}개)")
            return

        sequence_hours.sort()

        for pollutant_name, model in models_dict.items():
            values = []
            for hour in sequence_hours:
                v = hourly_data[hour]
                if pollutant_name.lower() == "pm10":
                    mean_val = np.mean([x[0] for x in v])
                elif pollutant_name.lower() == "co2":
                    mean_val = np.mean([x[1] for x in v])
                elif pollutant_name.lower() == "tvoc":
                    mean_val = np.mean([x[2] for x in v])
                else:
                    continue
                values.append(normalize(mean_val, mins_dict[pollutant_name], maxs_dict[pollutant_name]))

            if len(values) < 5:
                print(f"⚠️ 시리얼번호 {serial_number}, {pollutant_name} 데이터 시퀀스 부족")
                continue

            seq = np.array(values[-5:]).reshape(1, 5, 1)
            X = seq[:, :-1, :]
            y_true = denormalize(seq[:, -1, 0], mins_dict[pollutant_name], maxs_dict[pollutant_name])
            y_pred = denormalize(float(model.predict(X)), mins_dict[pollutant_name], maxs_dict[pollutant_name])

            error = abs(y_true - y_pred)
            threshold = thresholds_abs[pollutant_name.lower()]

            if error > threshold:
                anomaly = {
                    "sensorSerialNumber": serial_number,
                    "anomalyTimestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "pollutant": pollutant_name.upper(),
                    "pollutantValue": round(y_true, 2),
                    "predictedValue": round(y_pred, 2)
                }
                print(f"🚨 이상치 감지 (센서 {serial_number}, {pollutant_name}): {json.dumps(anomaly, indent=2)}")

                post_resp = requests.post(f"{BASE_URL}{outlier_endpoint}", json=anomaly, headers=headers)
                if post_resp.status_code == 200:
                    print(f"✅ 이상치 전송 성공 (센서 {serial_number}, {pollutant_name})")
                else:
                    print(f"🚫 이상치 전송 실패 (센서 {serial_number}, {pollutant_name}): {post_resp.status_code}, {post_resp.text}")
            else:
                print(f"ℹ️ 이상치 없음 (센서 {serial_number}, {pollutant_name}) 예측값: {round(y_pred, 2)}, 실제값: {round(y_true, 2)}")

    except Exception as e:
        print(f"⚠️ 오류 발생 (센서 {serial_number}): {e}")

# 실행 예제 (serial_numbers 리스트, models_dict, mins_dict, maxs_dict, token 모두 준비되어 있다고 가정)
models_dict = {
    "pm10": pm10_model,
    "co2": co2_model,
    "tvoc": tvoc_model
}

mins_dict = {
    "pm10": pm10_min,
    "co2": co2_min,
    "tvoc": tvoc_min
}

maxs_dict = {
    "pm10": pm10_max,
    "co2": co2_max,
    "tvoc": tvoc_max
}

for serial in serial_numbers:
    process_sensor_anomalies_and_post(bucket_name, serial, s3, 
                                      models_dict, mins_dict, maxs_dict)