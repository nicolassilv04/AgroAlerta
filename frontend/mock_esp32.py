from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/dados")
def dados():
    return jsonify({
        "pH": 6.2,
        "teor_umildade_%": 34,
        "nitrogenio_N_ppm": 120,
        "fosforo_P_ppm": 45,
        "potassio_K_ppm": 200,
        "condutividade_eletrica_dS_m": 1.8,
        "temperatura_solo": 26.5,
        "compactacao_solo_kPa": 3.1
    })

if __name__ == "__main__":
    app.run(port=5000)