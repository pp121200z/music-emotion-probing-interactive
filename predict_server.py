from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from src.config import get_config
from src.models.timesfm_extractor import TimesFMFeatureExtractor


app = Flask(__name__)
from flask_cors import CORS
CORS(app)

config = get_config()
print("used bio signals:", config.dataset.modalities)
print("config content:", config.__dict__)


config.timesfm.selected_layers = ['block_6', 'block_7']
config.timesfm.output_dim = 3840

feature_extractor = TimesFMFeatureExtractor(
    config=config,
    capture_layers=False
)

print("extractor actually using layer:", feature_extractor.selected_layers)

valence_model = joblib.load("results/valence_probe_model.joblib")
arousal_model = joblib.load("results/arousal_probe_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    print("config.dataset.modalities =", config.dataset.modalities)
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        
        class Signal:pass
        signal = Signal()

        modalities = config.dataset.modalities

        for modality in modalities:
            if modality in df.columns:
                setattr(signal, modality, df[modality].values)
                print(f" set up {modality}")
            else:
                setattr(signal, modality, np.zeros(128))
                print(f" lacking {modality},filled with 0")
                
        if 'ibi' not in modalities:
         setattr(signal, 'ibi', np.zeros(128))
         print("add dummy ibi(not in model, but extractor may require)")
        
        embedding_dict = feature_extractor.encode(signal.bvp)
        embedding = list(embedding_dict.values())[0].reshape(1, -1)

        val = valence_model.predict(embedding)[0]
        aro = arousal_model.predict(embedding)[0]

        return jsonify({"valence": int(val), "arousal": int(aro)})

    except Exception as e:
        print("Backend operation error", str(e))
        return jsonify({"error": str(e)}), 500
        s
        

if __name__ == "__main__":
    app.run(debug=True)
