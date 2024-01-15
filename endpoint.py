from flask import Flask, jsonify
from flask_cors import CORS
import csv


app = Flask(__name__)
CORS(app)


def read_csv(csv_file_path):
    payload = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader, None)

        for row in csv_reader:
            date = row[0]
            number = row[1]
            payload.append((date, number))
    return payload


@app.route('/api/data', methods=['GET'])
def receive_data():
    data = dict(read_csv('./new_case.csv'))
    print(f"Received data: {data}")
    return jsonify(data)


if __name__ == '__main__':
    app.run(port=5000)  # Enable debug mode for development
