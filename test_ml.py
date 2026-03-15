"""
MQTT-based ML Server for Warehouse Environmental Monitoring & Prediction
Loads an LSTM model (model_gudang_lstm.h5) and scaler (scaler_gudang) to provide
15-minute forecasts for temperature, humidity, and CO2 levels.

Communication is via MQTT (publish/subscribe) instead of HTTP.
- Subscribes to: synergy/ml/predict/request (single) & synergy/ml/predict/batch-request (batch)
- Publishes to:  synergy/ml/predict/response/{deviceId} & synergy/ml/predict/batch-response
- Status topic:  synergy/ml/status (retained, with LWT for offline detection)
"""

import os
import sys
import json
import signal
import warnings
import logging
import ssl
from datetime import datetime

# Set Keras backend to JAX (TensorFlow doesn't support Python 3.14+)
os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np
import joblib
import paho.mqtt.client as paho_mqtt

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(os.path.dirname(__file__), 'model', 'model_v2.keras')
)
SCALER_PATH = os.environ.get(
    'SCALER_PATH',
    os.path.join(os.path.dirname(__file__), 'model', 'scaler_v2')
)

# MQTT Broker
MQTT_HOST = os.environ.get('MQTT_HOST', 'mfe19520.ala.asia-southeast1.emqxsl.com')
MQTT_PORT = int(os.environ.get('MQTT_PORT', '8883'))
MQTT_USERNAME = os.environ.get('MQTT_USERNAME', 'backend-subscriber')
MQTT_PASSWORD = os.environ.get('MQTT_PASSWORD', 'Zufar123')
MQTT_CLIENT_ID = os.environ.get('MQTT_CLIENT_ID', 'synergy-ml-server-test')

# MQTT Topics
PREDICT_REQUEST_TOPIC = 'synergy/ml/predict/request'
PREDICT_RESPONSE_TOPIC = 'synergy/ml/predict/response'  # /{deviceId} appended
BATCH_REQUEST_TOPIC = 'synergy/ml/predict/batch-request'
BATCH_RESPONSE_TOPIC = 'synergy/ml/predict/batch-response'
ML_STATUS_TOPIC = 'synergy/ml/status'

# Scaler features: [temperature, humidity, co2, hour, feature4(0), feature5(0/1)]
# The model was trained with 6 features. Feature 4 is always 0, feature 5 is binary.
NUM_FEATURES = 6
SEQUENCE_LENGTH = 60  # Number of timesteps for LSTM input (60 minutes of 1-min data)

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Load Model & Scaler
# ============================================================================

model = None
scaler = None


def load_model_and_scaler():
    """Load the LSTM model and MinMaxScaler at startup."""
    global model, scaler

    logger.info(f'Loading scaler from: {SCALER_PATH}')
    scaler = joblib.load(SCALER_PATH)
    logger.info(f'Scaler loaded. Features: {scaler.n_features_in_}')
    logger.info(f'  Data min: {scaler.data_min_}')
    logger.info(f'  Data max: {scaler.data_max_}')

    logger.info(f'Loading LSTM model from: {MODEL_PATH}')

    # Try tensorflow first, fall back to keras
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info('Model loaded with TensorFlow/Keras')
    except ImportError:
        try:
            import keras
            model = keras.models.load_model(MODEL_PATH, compile=False)
            logger.info('Model loaded with standalone Keras (JAX backend)')
        except ImportError:
            logger.error(
                'Neither tensorflow nor keras is installed! '
                'Install with: pip install tensorflow or pip install keras'
            )
            raise

    # Log model info
    if hasattr(model, 'input_shape'):
        logger.info(f'  Input shape: {model.input_shape}')
    if hasattr(model, 'output_shape'):
        logger.info(f'  Output shape: {model.output_shape}')

    logger.info('Model and scaler loaded successfully!')


# ============================================================================
# ML Helpers
# ============================================================================


def prepare_input(sequence, current_hour=None):
    """
    Prepare input sequence for the LSTM model.

    Args:
        sequence: list of dicts with keys: temperature, humidity, co2
        current_hour: hour of day (0-23), defaults to current hour

    Returns:
        numpy array shaped (1, SEQUENCE_LENGTH, NUM_FEATURES) - scaled
    """
    if current_hour is None:
        current_hour = datetime.now().hour

    # Build feature array: [temp, humidity, co2, hour, 0, 0]
    raw_data = []
    for i, reading in enumerate(sequence):
        row = [
            float(reading['temperature']),
            float(reading['humidity']),
            float(reading['co2']),
            float(current_hour),
            0.0,  # Feature 4 (always 0 in training data)
            0.0   # Feature 5 (default 0, binary flag)
        ]
        raw_data.append(row)

    raw_array = np.array(raw_data, dtype=np.float32)

    # Scale using the fitted MinMaxScaler
    scaled = scaler.transform(raw_array)

    # Reshape for LSTM: (1, timesteps, features)
    return scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)


def inverse_transform_prediction(scaled_pred):
    """
    Inverse-transform the model's scaled output back to original values.
    The model outputs 3 values (temp, humidity, co2) or NUM_FEATURES values.
    """
    # If model outputs fewer features than scaler expects, pad with zeros
    if len(scaled_pred.shape) == 1:
        scaled_pred = scaled_pred.reshape(1, -1)

    n_out = scaled_pred.shape[1]

    if n_out < NUM_FEATURES:
        # Pad with zeros for the missing features
        padded = np.zeros((1, NUM_FEATURES), dtype=np.float32)
        padded[0, :n_out] = scaled_pred[0]
        inverse = scaler.inverse_transform(padded)
        return inverse[0, :n_out]
    else:
        inverse = scaler.inverse_transform(scaled_pred)
        return inverse[0]


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

    # Use only the last SEQUENCE_LENGTH readings
    sequence = sequence[-SEQUENCE_LENGTH:]

    # Validate data
    for i, reading in enumerate(sequence):
        for key in ['temperature', 'humidity', 'co2']:
            if key not in reading:
                return {
                    'device_id': device_id,
                    'error': f'Missing key "{key}" in reading {i}'
                }

    # Prepare input
    current_hour = datetime.now().hour
    model_input = prepare_input(sequence, current_hour)

    logger.info(
        f'Predicting for device {device_id}, '
        f'input shape: {model_input.shape}, '
        f'last reading: T={sequence[-1]["temperature"]}, '
        f'H={sequence[-1]["humidity"]}, CO2={sequence[-1]["co2"]}'
    )

    # Run inference
    prediction_scaled = model.predict(model_input, verbose=0)

    # Inverse transform
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

        # Subscribe to prediction request topics
        client.subscribe(PREDICT_REQUEST_TOPIC, qos=1)
        client.subscribe(BATCH_REQUEST_TOPIC, qos=1)
        logger.info(f'📥 Subscribed to: {PREDICT_REQUEST_TOPIC}')
        logger.info(f'📥 Subscribed to: {BATCH_REQUEST_TOPIC}')

        # Publish online status (retained so new subscribers see it immediately)
        status_payload = json.dumps({
            'status': 'online',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'timestamp': datetime.now().isoformat()
        })
        #client.publish(ML_STATUS_TOPIC, status_payload, qos=1, retain=True)
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
            {"temperature": 30.5, "humidity": 75.2, "co2": 450},
            ... (10 readings)
        ]
    }

    Publishes result to: synergy/ml/predict/response/{device_id}
    """
    device_id = data.get('device_id', 'unknown')
    sequence = data.get('sequence', [])

    result = run_prediction(device_id, sequence)

    # Publish response to device-specific topic
    response_topic = f'{PREDICT_RESPONSE_TOPIC}/{device_id}'
    #client.publish(response_topic, json.dumps(result), qos=1)
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

    Publishes results to: synergy/ml/predict/batch-response
    """
    devices = data.get('devices', [])
    results = []

    for device_data in devices:
        device_id = device_data.get('device_id', 'unknown')
        sequence = device_data.get('sequence', [])
        result = run_prediction(device_id, sequence)
        results.append(result)

    #client.publish(BATCH_RESPONSE_TOPIC, json.dumps({'results': results}), qos=1)
    logger.info(f'📤 Published batch prediction ({len(results)} devices) to: {BATCH_RESPONSE_TOPIC}')


# ============================================================================
# Main
# ============================================================================


def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info('\n🛑 Shutting down ML server...')
    # Publish offline status before disconnecting
    if mqtt_client and mqtt_client.is_connected():
        #mqtt_client.publish(ML_STATUS_TOPIC, json.dumps({
        #    'status': 'offline',
        #    'timestamp': datetime.now().isoformat()
        #}), qos=1, retain=True)
        mqtt_client.disconnect()
    sys.exit(0)


mqtt_client = None

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

    # Last Will and Testament — broker publishes this if we disconnect unexpectedly
    #lwt_payload = json.dumps({
    #    'status': 'offline',
    #    'reason': 'unexpected_disconnect',
    #    'timestamp': datetime.now().isoformat()
    #})
    #mqtt_client.will_set(ML_STATUS_TOPIC, lwt_payload, qos=1, retain=True)

    # Register callbacks
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message = on_message

    # Connect and start the loop (blocking — handles reconnection automatically)
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)

    logger.info('🚀 ML server started. Entering MQTT loop...\n')
    mqtt_client.loop_forever()
