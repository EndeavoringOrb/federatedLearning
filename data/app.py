from flask import Flask, render_template, jsonify, request
import os
import json
import pandas as pd

app = Flask(__name__)
RUNS_DIR = "data/tool/trainingRuns"


@app.route("/")
def index():
    runs = sorted(
        [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.isdigit()]
    )
    return render_template("index.html", runs=runs)


@app.route("/get_run_data", methods=["POST"])
def get_run_data():
    run_ids = request.json.get("runs", [])
    result = {}
    for run_id in run_ids:
        path = os.path.join(RUNS_DIR, run_id)
        try:
            with open(os.path.join(path, "config.json")) as f:
                config = json.load(f)
            df = pd.read_csv(os.path.join(path, "loss.csv"))
            result[run_id] = {"config": config, "loss": df.to_dict(orient="list")}
        except Exception as e:
            result[run_id] = {"error": str(e)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
