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

# 유틸 함수
def normalize(value, min_v, max_v):
    return (value - min_v) / (max_v - min_v)

def denormalize(value, min_v, max_v):
    return value * (max_v - min_v) + min_v

def convert_to_float(value):
    if isinstance(value, np.float32):
        return float(value)
    return value

if __name__ == "__main__":
    # 센서 ID 가져오기
    getid_endpoint = "/sensorMappingWithRoom"
    try:
        response = requests.get(f"{BASE_URL}{getid_endpoint}")
        if response.status_code == 200:
            print("🚀 센서 응답:", response.text)
            data = response.json()
            serial_numbers = [sensor["sensorSerialNumber"] for sensor in data]
            print("Sensor Serial Numbers:", serial_numbers)
        else:
            print(f"🚫 센서 조회 실패 ({response.status_code}): {response.text}")
            exit()
    except Exception as e:
        print(f"⚠️ 센서 요청 실패: {e}")
        exit()

    predict_endpoint = "/predictedAirQuality"

    def predict_from_multiple_files(s3, serial_number: int,
                                pm10_min, pm10_max,
                                co2_min, co2_max,
                                tvoc_min, tvoc_max):
        try:
            prefix = f"airQuality/{serial_number}/"
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=2500)
            files = response.get('Contents', [])

            if not files:
                print("⚠️ 파일이 없습니다.")
                return []

            # 최신 수정 시간 기준 정렬 (최근 데이터 우선)
            files.sort(key=lambda x: x['LastModified'], reverse=True)

            # 최근 파일만 추출
            files = files[:2000] 

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

                    # ✅ 5시간치 확보되면 중단
                    if len(hourly_data) >= 5:
                        break

                except Exception as e:
                    print(f"⚠️ 파일 처리 오류 ({file['Key']}): {e}")

            sorted_hours = sorted(hourly_data.keys(), reverse=True)
            sequence_hours = sorted(sorted_hours[:5])  # 최근 5시간 사용

            print("🕒 예측에 사용된 시간대:")
            for h in sequence_hours:
                print(" -", h.strftime("%Y-%m-%dT%H:%M:%S"))

            if len(sequence_hours) < 5:
                print(f"⚠️ 시간대 시퀀스 부족 (현재 {len(sequence_hours)}개)")
                return []

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
                input_pm10 = np.array(pm10_seq[-5:]).reshape(1, 5, 1)
                input_co2 = np.array(co2_seq[-5:]).reshape(1, 5, 1)
                input_tvoc = np.array(tvoc_seq[-5:]).reshape(1, 5, 1)

                try:
                    pred_pm10 = pm10_model.predict(input_pm10)[0][0]
                    pred_co2 = co2_model.predict(input_co2)[0][0]
                    pred_tvoc = tvoc_model.predict(input_tvoc)[0][0]
                except Exception as e:
                    print(f"⚠️ 예측 오류: {e}")
                    break

                denorm_pm10 = round(convert_to_float(denormalize(pred_pm10, pm10_min, pm10_max)), 2)
                denorm_co2 = round(convert_to_float(denormalize(pred_co2, co2_min, co2_max)), 2)
                denorm_tvoc = round(convert_to_float(denormalize(pred_tvoc, tvoc_min, tvoc_max)), 2)

                predictions.append({
                    "sensorSerialNumber": serial_number,
                    "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
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

    # 모든 센서에 대해 예측 및 전송
    try:
        all_predictions = []
        for serial_number in serial_numbers:
            preds = predict_from_multiple_files(
                s3,
                serial_number,
                pm10_min, pm10_max,
                co2_min, co2_max,
                tvoc_min, tvoc_max
            )
            all_predictions.extend(preds)

        if not all_predictions:
            print("⚠️ 예측 결과 없음, 전송 생략")
            exit()

        print("📦 예측 데이터:")
        print(json.dumps(all_predictions, indent=4))

        headers = {
            'accept': '*/*',
            'Content-Type': "application/json"
        }

        response = requests.post(
            f"{BASE_URL}{predict_endpoint}",
            json=all_predictions,
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            print("✅ 예측 데이터 전송 성공")
            print("📬 응답 코드:", response.status_code)
            print("📬 응답 내용:", response.text.strip())
        else:
            print(f"🚫 전송 실패 ({response.status_code}): {response.text.strip()}")

    except Exception as e:
        print(f"⚠️ 예측 전송 실패: {e}")
