from fastapi import FastAPI
from fastapi.responses import JSONResponse
import boto3
import json
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import models
from keras import layers
from tensorflow.python.keras.models import load_model
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import requests
from collections import defaultdict
from dotenv import load_dotenv
import os
BASE_URL = "https://smartair.site"

app = FastAPI()
load_dotenv(dotenv_path="key.env")

# S3 접근 정보
access_key_id = os.getenv("ACCESS_KEY_ID")
secret_access_key = os.getenv("SECRET_ACCESS_KEY")
region = 'ap-northeast-2'
bucket_name = os.getenv("BUCKET_NAME")
password = os.getenv("PASSWORD")

# 모델 로드
pm10_model = keras.models.load_model("pm10_model.h5", compile=False)
co2_model = keras.models.load_model("co2_model.h5", compile=False)
tvoc_model = keras.models.load_model("tvoc_model.h5", compile=False)

# 정규화 범위
pm10_min, pm10_max = 0.0, 35.0
co2_min, co2_max = 400.0, 1324.0
tvoc_min, tvoc_max = 0.0, 354.0

# S3 클라이언트 설정
s3 = boto3.client(
    's3',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name=region
)


login_endpoint = "/login"
# 로그인 데이터
login_data = {
    "email": "admin@example.com",
    "password": password
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


# 정규화 함수
def normalize(value, min_v, max_v):
    return (value - min_v) / (max_v - min_v)
def denormalize(value, min_v, max_v):
    return value * (max_v - min_v) + min_v
def convert_to_float(value):
    if isinstance(value, np.float32):
        return float(value)
    return value



predict_endpoint = "/predictedAirQuality"
# 예측 로직 (1시간 단위로 평균내어 예측)
def predict_from_multiple_files(s3, serial_number: int, 
                                pm10_min, pm10_max, 
                                co2_min, co2_max, 
                                tvoc_min, tvoc_max):
    try:
        prefix = f"airQuality/{serial_number}/"
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        files = response.get('Contents', [])

        if not files:
            print("⚠️ 파일이 없습니다.")
            return []

        # 파일명을 기준으로 시간 추출해 정렬
        files.sort(key=lambda x: x['Key'].split('/')[-1], reverse=True)
        files = files[:300]  

        # 시간대별로 그룹핑
        hourly_data = defaultdict(list)
        for file in files:
            try:
                key = file['Key']
                timestamp_str = key.split('/')[-1].replace('.json', '').split('.')[0]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)

                obj = s3.get_object(Bucket=bucket_name, Key=key)
                content = obj['Body'].read().decode('utf-8')
                data = json.loads(content)

                pm10_avg = np.mean([
                    data.get("pt1", {}).get("pm100_standard", 0),
                    data.get("pt2", {}).get("pm100_standard", 0)
                ])
                co2 = data.get("eco2", 0)
                tvoc = data.get("tvoc", 0)

                hourly_data[hour_key].append((pm10_avg, co2, tvoc))
            except Exception as e:
                print(f"⚠️ 파일 처리 오류 ({file['Key']}): {e}")

        # 최근 시각 기준 5개 시간대 평균값 확보
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
            return []

        sequence_hours.sort()

        pm10_seq, co2_seq, tvoc_seq = [], [], []
        for hour in sequence_hours:
            values = hourly_data[hour]
            pm10_mean = np.mean([v[0] for v in values])
            co2_mean = np.mean([v[1] for v in values])
            tvoc_mean = np.mean([v[2] for v in values])

            pm10_seq.append(normalize(pm10_mean, pm10_min, pm10_max))
            co2_seq.append(normalize(co2_mean, co2_min, co2_max))
            tvoc_seq.append(normalize(tvoc_mean, tvoc_min, tvoc_max))

        predictions = []
        current_time = sequence_hours[-1] + timedelta(hours=1)
        midnight = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        while current_time <= midnight:
            next_prediction_time = current_time

            input_pm10 = np.array(pm10_seq[-5:]).reshape(1, 5, 1)
            input_co2 = np.array(co2_seq[-5:]).reshape(1, 5, 1)
            input_tvoc = np.array(tvoc_seq[-5:]).reshape(1, 5, 1)

            try:
                pred_pm10 = pm10_model.predict(input_pm10)[0][0]
                pred_co2 = co2_model.predict(input_co2)[0][0]
                pred_tvoc = tvoc_model.predict(input_tvoc)[0][0]
            except Exception as e:
                print(f"⚠️ 모델 예측 오류: {e}")
                break

            denorm_pm10 = round(convert_to_float(denormalize(pred_pm10, pm10_min, pm10_max)), 2)
            denorm_co2 = round(convert_to_float(denormalize(pred_co2, co2_min, co2_max)), 2)
            denorm_tvoc = round(convert_to_float(denormalize(pred_tvoc, tvoc_min, tvoc_max)), 2)

            predictions.append({
                "sensorSerialNumber": serial_number,
                "timestamp": next_prediction_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "pm10": denorm_pm10,
                "co2": denorm_co2,
                "tvoc": denorm_tvoc
            })

            pm10_seq.append(normalize(denorm_pm10, pm10_min, pm10_max))
            co2_seq.append(normalize(denorm_co2, co2_min, co2_max))
            tvoc_seq.append(normalize(denorm_tvoc, tvoc_min, tvoc_max))

            current_time += timedelta(hours=1)

        return predictions

    except Exception as e:
        print(f"⚠️ 예측 로직 오류: {e}")
        return []




# 매핑된 모든 센서에 대해 예측 수행 및 전송
try:
    all_predictions = []
    for serial_number in serial_numbers:
        predictions = predict_from_multiple_files(
            s3,
            serial_number,
            pm10_min, pm10_max,
            co2_min, co2_max,
            tvoc_min, tvoc_max
        )
        all_predictions.extend(predictions)

    if not all_predictions:
        print("⚠️ 예측 결과가 비어 있습니다. 전송하지 않습니다.")
        exit()

    json_string = json.dumps(all_predictions, indent=4)
    print("🔄 예측 결과:", json_string)

    headers = {
        'accept': '*/*',
        'Content-Type': "application/json",
        'Authorization': f"Bearer {token}"
    }

    try:
        response = requests.post(
            f"{BASE_URL}{predict_endpoint}",
            json=all_predictions,
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            print("✅ 예측 데이터 전송 성공")
            print("🔄 응답 상태 코드:", response.status_code)
            print("🔄 응답 본문:", response.text.strip())
        else:
            print(f"🚫 서버 오류 ({response.status_code}):")
            print(response.text.strip())

    except requests.exceptions.Timeout:
        print("⚠️ 요청이 타임아웃되었습니다. 서버가 응답하지 않습니다.")

    except requests.exceptions.RequestException as e:
        print(f"⚠️ 네트워크 오류: {e}")

except Exception as e:
    print(f"⚠️ 예측 데이터 요청 실패: {e}")
