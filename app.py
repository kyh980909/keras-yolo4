from flask import Flask, render_template, jsonify
from flask import request
import count_coin
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/jpy', methods=['POST'])
def jpy():
    if request.method == 'POST':
        img = request.get_data()
        result = count_coin.jpy_count_coin(json.loads(img)['img'])
        return json.dumps(result)

@app.route('/krw', methods=['POST'])
def krw():
    if request.method == 'POST':
        img = request.get_data()
        result = count_coin.krw_count_coin(json.loads(img)['img'])
        return json.dumps(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)