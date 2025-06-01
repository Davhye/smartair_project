from fastapi import FastAPI
from fastapi.responses import JSONResponse
import boto3
import json
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from datetime import datetime, timedelta
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

# 모델 로드
pm10_model = keras.models.load_model("pm10_model.h5", compile=False)
co2_model = keras.models.load_model('co2_model.h5', compile=False)
tvoc_model = keras.models.load_model('tvoc_model.h5', compile=False)

# 정규화 범위
pm10_min, pm10_max = 0.0, 133.0
co2_min, co2_max = 400.0, 3229.0
tvoc_min, tvoc_max = 0.0, 500.0

def normalize(value, min_v, max_v):
    return (value - min_v) / (max_v - min_v)

def denormalize(value, min_v, max_v):
    return float(value * (max_v - min_v) + min_v)

def main():
    # 센서 아이디 받아오기 (토큰 없이)
    getid_endpoint = "/sensorMappingWithRoom"
    try:
        response = requests.get(f"{BASE_URL}{getid_endpoint}")
        if response.status_code == 200:
            print("🚀 응답 내용:", response.text)
            data = response.json()
            serial_numbers = [sensor["sensorSerialNumber"] for sensor in data]
            print("Sensor Serial Numbers:", serial_numbers)
        else:
            print(f"🚫 센서 조회 실패 ({response.status_code}): {response.text}")
            return
    except Exception as e:
        print(f"⚠️ 센서 요청 실패: {e}")
        return

    thresholds_abs = {
        "pm10": 13,
        "co2": 330,
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
        'Content-Type': "application/json"
        # Authorization 헤더 없이
    }

    outlier_endpoint = "/api/reports/anomaly"

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
            hourly_data = defaultdict(list)

            #시간 키 저장용 set 생성
            collected_hours = set()

            for file in files:
                try:
                    key = file['Key']
                    filename = key.split('/')[-1].replace('.json', '')
                    timestamp_str = filename.split('.')[0] 
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                    hour_key = timestamp.replace(minute=0, second=0, microsecond=0)

                    # 파일 내용 파싱
                    resp = s3_client.get_object(Bucket=bucket_name, Key=key)
                    content = resp['Body'].read().decode('utf-8')
                    data = json.loads(content)

                    pm10_avg = np.mean([
                        data.get("pt1", {}).get("pm100_standard", 0),
                        data.get("pt2", {}).get("pm100_standard", 0)
                    ])
                    co2 = data.get("eco2", 0)
                    tvoc = data.get("tvoc", 0)

                    hourly_data[hour_key].append((pm10_avg, co2, tvoc))
                    collected_hours.add(hour_key)

                    #6시간 분량 모이면 조기 종료
                    if len(collected_hours) >= 6:
                        break

                except Exception as e:
                    print(f"⚠️ 파일 처리 오류 ({file['Key']}): {e}")

            sorted_hours = sorted(hourly_data.keys(), reverse=True)
            sequence_hours = sorted(sorted_hours[:5])

            if len(sequence_hours) < 5:
                print(f"⚠️ 평균값 시퀀스 부족 (현재: {len(sequence_hours)}개)")
                return

            for pollutant_name, model in models_dict.items():
                values = []
                for hour in sequence_hours:
                    v = hourly_data[hour]
                    mean_val = {
                        "pm10": np.mean([x[0] for x in v]),
                        "co2": np.mean([x[1] for x in v]),
                        "tvoc": np.mean([x[2] for x in v])
                    }[pollutant_name.lower()]
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
                        "anomalyTimestamp": (sequence_hours[-1] + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                        "pollutant": pollutant_name.upper(),
                        "pollutantValue": round(y_true, 2),
                        "predictedValue": round(y_pred, 2)
                    }
                    print(f"🚨 이상치 감지 (센서 {serial_number}, {pollutant_name}): {json.dumps(anomaly, indent=2)}")
                    post_resp = requests.post(f"{BASE_URL}{outlier_endpoint}", json=anomaly, headers=headers)
                    if post_resp.status_code == 200:
                        print(f"✅ 이상치 전송 성공 (센서 {serial_number}, {pollutant_name})")
                    else:
                        print(f"🚫 이상치 전송 실패 ({post_resp.status_code}): {post_resp.text}")
                else:
                    print(f"ℹ️ 이상치 없음 (센서 {serial_number}, {pollutant_name}) 예측값: {round(y_pred, 2)}, 실제값: {round(y_true, 2)}")

        except Exception as e:
            print(f"⚠️ 오류 발생 (센서 {serial_number}): {e}")

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
        process_sensor_anomalies_and_post(bucket_name, serial, s3, models_dict, mins_dict, maxs_dict)

if __name__ == "__main__":
    main()
