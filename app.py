from flask import Flask, render_template, jsonify
from flask import request
import count_coin
import json
from detection import detect_japan_obj

app = Flask(__name__)

# @app.route('/jpy', methods=['POST'])
# def jpy():
#     if request.method == 'POST':
#         img = request.get_data()
#         result = count_coin.jpy_count_coin(json.loads(img)['img'])
#         return json.dumps(result)


@app.route('/jpy', methods=['POST'])
def jpy():
    if request.method == 'POST':
        img = json.loads(request.get_data())['img']
        result = detect_japan_obj.detect(img)
        return json.dumps(result)

if __name__ == "__main__":
    _decode = count_coin.init()
    app.run(host='0.0.0.0', debug=True)