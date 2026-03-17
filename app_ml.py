"""
MQTT-based ML Server for Warehouse Environmental Monitoring & Prediction
Loads an LSTM model (model_v3.keras) and scaler (scaler_v3.gz) to provide
15-minute forecasts for temperature, humidity, and CO2 levels.

Communication is via MQTT (publish/subscribe) instead of HTTP.
- Subscribes to: synergy/ml/predict/request (single) & synergy/ml/predict/batch-request (batch)
- Publishes to:  synergy/ml/predict/response/{deviceId} & synergy/ml/predict/batch-response
- Status topic:  synergy/ml/status (retained, with LWT for offline detection)

Training features (7 total):
  [suhu, kelembapan, co2, hour_sin, hour_cos, status_kipas, status_dehumidifier]

Model input shape : (1, 240, 7)  — 240 timesteps × 7 features
Model output shape: (1, 3)       — [suhu, kelembapan, co2] predicted 15 min ahead
"""

import os
import sys
import json
import signal
import warnings
import logging
import ssl
from datetime import datetime, timedelta

import numpy as np
import joblib
import paho.mqtt.client as paho_mqtt
import tensorflow as tf

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# Patch untuk Keras Version Mismatch
# ============================================================================
# Model disimpan di Colab (Keras baru) yang menyimpan 'quantization_config'
# di config Dense layer. Tapi Keras di VM lebih lama dan tidak mengenal
# parameter itu → crash saat load_model.
#
# Solusi: monkey-patch Dense.__init__ LANGSUNG agar membuang parameter tsb.
# Ini bekerja karena Keras deserializer menggunakan class Dense yang sama.
import keras.layers
_original_dense_init = keras.layers.Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)

keras.layers.Dense.__init__ = _patched_dense_init

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# --- Path ke file model & scaler ---
# Sesuaikan dengan nama file hasil training di Colab
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(os.path.dirname(__file__), 'model', 'model_v3.keras')
)
SCALER_PATH = os.environ.get(
    'SCALER_PATH',
    os.path.join(os.path.dirname(__file__), 'model', 'scaler_v3')
)

# MQTT Broker
MQTT_HOST = os.environ.get('MQTT_HOST', 'mfe19520.ala.asia-southeast1.emqxsl.com')
MQTT_PORT = int(os.environ.get('MQTT_PORT', '8883'))
MQTT_USERNAME = os.environ.get('MQTT_USERNAME', 'backend-subscriber')
MQTT_PASSWORD = os.environ.get('MQTT_PASSWORD', 'Zufar123')
MQTT_CLIENT_ID = os.environ.get('MQTT_CLIENT_ID', 'synergy-ml-server-v3-testing')

# MQTT Topics
PREDICT_REQUEST_TOPIC = 'synergy/ml/predict/request'
PREDICT_RESPONSE_TOPIC = 'synergy/ml/predict/response'  # /{deviceId} appended
BATCH_REQUEST_TOPIC = 'synergy/ml/predict/batch-request'
BATCH_RESPONSE_TOPIC = 'synergy/ml/predict/batch-response'
ML_STATUS_TOPIC = 'synergy/ml/status'

# --- Model / Scaler Constants ---
# Training menggunakan 7 fitur: suhu, kelembapan, co2, hour_sin, hour_cos,
#                                 status_kipas, status_dehumidifier
NUM_FEATURES = 7
SEQUENCE_LENGTH = 240  # 1 jam data historis (240 titik × 15 detik)
SAMPLING_INTERVAL_SEC = 15  # interval antar data point

# ============================================================================
# Load Model & Scaler
# ============================================================================

model = None
scaler = None


def load_model_and_scaler():
    """Load the LSTM model and MinMaxScaler at startup."""
    global model, scaler

    # --- Load Scaler ---
    logger.info(f'Loading scaler from: {SCALER_PATH}')
    scaler = joblib.load(SCALER_PATH)
    logger.info(f'Scaler loaded. Features: {scaler.n_features_in_}')
    logger.info(f'  Data min: {scaler.data_min_}')
    logger.info(f'  Data max: {scaler.data_max_}')

    # --- Load Model ---
    logger.info(f'Loading LSTM model from: {MODEL_PATH}')
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    logger.info(f'✅ Model loaded successfully!')

    if hasattr(model, 'input_shape'):
        logger.info(f'  Input shape : {model.input_shape}')
    if hasattr(model, 'output_shape'):
        logger.info(f'  Output shape: {model.output_shape}')


# ============================================================================
# ML Helpers
# ============================================================================


def prepare_input(sequence):
    """
    Prepare input sequence for the LSTM model.

    Setiap reading HARUS memiliki timestamp sendiri (atau minimal kita
    hitung berdasarkan urutan). Ini penting karena hour_sin/hour_cos
    berbeda untuk setiap data point — sama seperti saat training.

    Args:
        sequence: list of dicts with keys:
            - temperature (float)
            - humidity (float)
            - co2 (float)
            - Optional: timestamp (ISO string), status_kipas (0/1),
              status_dehumidifier (0/1)

    Returns:
        numpy array shaped (1, SEQUENCE_LENGTH, NUM_FEATURES) — scaled
    """
    raw_data = []

    for i, reading in enumerate(sequence):
        # --- Hitung hour_sin & hour_cos per-reading ---
        if 'timestamp' in reading:
            # Jika ada timestamp di payload, gunakan itu
            ts = datetime.fromisoformat(reading['timestamp'])
        else:
            # Fallback: asumsikan reading terakhir = sekarang,
            # yang sebelumnya mundur per SAMPLING_INTERVAL_SEC
            now = datetime.now()
            offset = (len(sequence) - 1 - i) * SAMPLING_INTERVAL_SEC
            ts = now - timedelta(seconds=offset)

        # Fractional hour: 14:30:15 → 14.504167
        hour_frac = ts.hour + (ts.minute / 60.0) + (ts.second / 3600.0)
        hour_sin = np.sin(2 * np.pi * hour_frac / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_frac / 24.0)

        # Baca status kipas & dehumidifier dari payload jika ada
        kipas = float(reading.get('status_kipas', 0))
        dehumidifier = float(reading.get('status_dehumidifier', 0))

        row = [
            float(reading['temperature']),   # 0: suhu
            float(reading['humidity']),       # 1: kelembapan
            float(reading['co2']),            # 2: co2
            float(hour_sin),                  # 3: hour_sin
            float(hour_cos),                  # 4: hour_cos
            kipas,                            # 5: status_kipas
            dehumidifier                      # 6: status_dehumidifier
        ]
        raw_data.append(row)

    raw_array = np.array(raw_data, dtype=np.float32)

    # Scale menggunakan MinMaxScaler yang sama dengan training
    scaled = scaler.transform(raw_array)

    # Reshape untuk LSTM: (1, timesteps, features)
    return scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)


def inverse_transform_prediction(scaled_pred):
    """
    Inverse-transform output model yang ter-scaled kembali ke nilai asli.
    Model mengeluarkan 3 nilai: [suhu, kelembapan, co2] (ter-scaled).
    Kita perlu memasukkannya ke posisi yang benar di array 7-fitur
    sebelum inverse transform.
    """
    if len(scaled_pred.shape) == 1:
        scaled_pred = scaled_pred.reshape(1, -1)

    n_out = scaled_pred.shape[1]  # seharusnya 3

    # Buat array dummy 7-fitur, isi posisi [0,1,2] dengan output model
    padded = np.zeros((1, NUM_FEATURES), dtype=np.float32)
    padded[0, :n_out] = scaled_pred[0]

    inverse = scaler.inverse_transform(padded)
    return inverse[0, :n_out]  # ambil hanya suhu, kelembapan, co2


def run_prediction(device_id, sequence):
    """
    Run a single prediction for a device.
    Returns dict with prediction results or error.
    """
    if len(sequence) < SEQUENCE_LENGTH:
        return {
            'device_id': device_id,
            'error': f'Need at least {SEQUENCE_LENGTH} readings, got {len(sequence)}'
        }

    # Gunakan hanya 240 data terakhir
    sequence = sequence[-SEQUENCE_LENGTH:]

    # Validasi data
    for i, reading in enumerate(sequence):
        for key in ['temperature', 'humidity', 'co2']:
            if key not in reading:
                return {
                    'device_id': device_id,
                    'error': f'Missing key "{key}" in reading {i}'
                }

    # Prepare input (hour_sin/cos dihitung per-reading di dalam prepare_input)
    model_input = prepare_input(sequence)

    logger.info(
        f'Predicting for device {device_id}, '
        f'input shape: {model_input.shape}, '
        f'last reading: T={sequence[-1]["temperature"]}, '
        f'H={sequence[-1]["humidity"]}, CO2={sequence[-1]["co2"]}'
    )

    # Run inference
    prediction_scaled = model.predict(model_input, verbose=0)

    # Inverse transform ke nilai asli
    prediction = inverse_transform_prediction(prediction_scaled[0])

    result = {
        'device_id': device_id,
        'predicted_temperature': round(float(prediction[0]), 2),
        'predicted_humidity': round(float(prediction[1]), 2),
        'predicted_co2': round(float(prediction[2]), 2),
        'prediction_horizon_min': 15,
        'timestamp': datetime.now().isoformat()
    }

    logger.info(
        f'Prediction result: T={result["predicted_temperature"]}°C, '
        f'H={result["predicted_humidity"]}%, '
        f'CO2={result["predicted_co2"]}ppm'
    )

    return result


# ============================================================================
# MQTT Callbacks
# ============================================================================


def on_connect(client, userdata, connect_flags, reason_code, properties=None):
    """Called when the client connects to the MQTT broker."""
    if reason_code == 0 or str(reason_code) == 'Success':
        logger.info('✅ Connected to MQTT broker')

        client.subscribe(PREDICT_REQUEST_TOPIC, qos=1)
        client.subscribe(BATCH_REQUEST_TOPIC, qos=1)
        logger.info(f'📥 Subscribed to: {PREDICT_REQUEST_TOPIC}')
        logger.info(f'📥 Subscribed to: {BATCH_REQUEST_TOPIC}')

        # Publish online status
        status_payload = json.dumps({
            'status': 'online',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'timestamp': datetime.now().isoformat()
        })
        client.publish(ML_STATUS_TOPIC, status_payload, qos=1, retain=True)
        logger.info(f'📡 Published online status to {ML_STATUS_TOPIC}')
        logger.info('⏳ Waiting for prediction requests...\n')
    else:
        logger.error(f'❌ Connection failed: {reason_code}')


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    """Called when the client disconnects from the MQTT broker."""
    logger.warning(f'🔌 Disconnected from broker (reason: {reason_code}). Reconnecting...')


def on_message(client, userdata, message):
    """Called when a message is received on a subscribed topic."""
    topic = message.topic
    try:
        payload = json.loads(message.payload.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f'❌ Invalid message payload on {topic}: {e}')
        return

    logger.info(f'📨 Received message on: {topic}')

    try:
        if topic == PREDICT_REQUEST_TOPIC:
            handle_predict_request(client, payload)
        elif topic == BATCH_REQUEST_TOPIC:
            handle_batch_predict_request(client, payload)
        else:
            logger.warning(f'⚠️ Unhandled topic: {topic}')
    except Exception as e:
        logger.error(f'❌ Error processing message on {topic}: {e}', exc_info=True)


# ============================================================================
# Request Handlers
# ============================================================================


def handle_predict_request(client, data):
    """
    Handle a single prediction request.

    Expected payload:
    {
        "device_id": "uuid",
        "sequence": [
            {
                "temperature": 30.5,
                "humidity": 75.2,
                "co2": 450,
                "timestamp": "2026-03-17T14:30:00",  // optional
                "status_kipas": 0,                     // optional, default 0
                "status_dehumidifier": 1               // optional, default 0
            },
            ... (240 readings)
        ]
    }
    """
    device_id = data.get('device_id', 'unknown')
    sequence = data.get('sequence', [])

    result = run_prediction(device_id, sequence)

    response_topic = f'{PREDICT_RESPONSE_TOPIC}/{device_id}'
    client.publish(response_topic, json.dumps(result), qos=1)
    logger.info(f'📤 Published prediction to: {response_topic}')


def handle_batch_predict_request(client, data):
    """
    Handle a batch prediction request for multiple devices.

    Expected payload:
    {
        "devices": [
            { "device_id": "uuid1", "sequence": [...] },
            { "device_id": "uuid2", "sequence": [...] }
        ]
    }
    """
    devices = data.get('devices', [])
    results = []

    for device_data in devices:
        device_id = device_data.get('device_id', 'unknown')
        sequence = device_data.get('sequence', [])
        result = run_prediction(device_id, sequence)
        results.append(result)

    client.publish(BATCH_RESPONSE_TOPIC, json.dumps({'results': results}), qos=1)
    logger.info(f'📤 Published batch prediction ({len(results)} devices) to: {BATCH_RESPONSE_TOPIC}')


# ============================================================================
# Main
# ============================================================================

mqtt_client = None


def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info('\n🛑 Shutting down ML server...')
    if mqtt_client and mqtt_client.is_connected():
        mqtt_client.publish(ML_STATUS_TOPIC, json.dumps({
            'status': 'offline',
            'timestamp': datetime.now().isoformat()
        }), qos=1, retain=True)
        mqtt_client.disconnect()
    sys.exit(0)


if __name__ == '__main__':
    # Load ML model and scaler first
    load_model_and_scaler()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # Create MQTT client
    logger.info(f'\n📡 Connecting to MQTT broker: {MQTT_HOST}:{MQTT_PORT}')
    logger.info(f'   Client ID: {MQTT_CLIENT_ID}')
    logger.info(f'   Username:  {MQTT_USERNAME}')

    mqtt_client = paho_mqtt.Client(
        callback_api_version=paho_mqtt.CallbackAPIVersion.VERSION2,
        client_id=MQTT_CLIENT_ID,
        protocol=paho_mqtt.MQTTv311
    )

    # Authentication
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    # TLS for EMQX Cloud (port 8883)
    mqtt_client.tls_set(tls_version=ssl.PROTOCOL_TLS)

    # Last Will and Testament
    lwt_payload = json.dumps({
        'status': 'offline',
        'reason': 'unexpected_disconnect',
        'timestamp': datetime.now().isoformat()
    })
    mqtt_client.will_set(ML_STATUS_TOPIC, lwt_payload, qos=1, retain=True)

    # Register callbacks
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message = on_message

    # Connect and start the loop
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)

    logger.info('🚀 ML server started. Entering MQTT loop...\n')
    mqtt_client.loop_forever()
