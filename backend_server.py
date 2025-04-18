from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import csv
import os

app = Flask(__name__)
CORS(app)

# To store latest vehicle count in memory
latest_count = 0

# CSV file to log data
csv_file = "vehicle_counts_log.csv"

# Create CSV file with header if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "vehicle_count"])

@app.route('/vehicle_count', methods=['POST'])
def receive_vehicle_count():
    global latest_count
    data = request.json
    count = data.get("count", 0)
    latest_count = count

    #Log count to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), count])

    return jsonify({"status": "success", "message": "Count received"}), 200

@app.route('/current_count', methods=['GET'])
def get_current_count():
    return jsonify({"count": latest_count})

if __name__ == '__main__':
    app.run(debug=True)
