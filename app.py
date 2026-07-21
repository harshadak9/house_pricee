import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Model path
MODEL_PATH = "Linear_Reggressionmodel.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: `{MODEL_PATH}` not found in root directory!")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"❌ Failed to load pickle model: {e}")
        return None

model = load_model()

# Embedded Single-Page Application (HTML + CSS Animations + JS AJAX)
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor (Linear Regression)</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #032b42 100%);
            --card-bg: rgba(255, 255, 255, 0.06);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent-color: #38bdf8;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--text-main);
            padding: 30px 20px;
        }

        /* Glassmorphism Card Container */
        .container {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 950px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        header p {
            color: var(--text-sub);
            font-size: 0.95rem;
            margin-top: 5px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-sub);
        }

        input[type="number"], select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 12px 16px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.3s ease;
        }

        input[type="number"]:focus, select:focus {
            border-color: var(--accent-color);
        }

        /* Submit Button Styling */
        .btn-submit {
            margin-top: 30px;
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(90deg, #0284c7, #6366f1);
            color: #fff;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(2, 132, 199, 0.4);
            transition: all 0.3s ease;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
        }

        .btn-submit:active {
            transform: translateY(1px);
        }

        /* Prediction Result Animation Box */
        .result-box {
            margin-top: 25px;
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            display: none;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-box.success {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #38bdf8;
        }

        .result-box.error {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
        }

        .result-box h2 {
            font-size: 2rem;
            margin-top: 5px;
        }

        /* Spinner Animation */
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <div class="container">
        <header>
            <h1>🏡 House Price Predictor</h1>
            <p>Linear Regression Model Engine</p>
        </header>

        <form id="predictionForm">
            <div class="form-grid">
                
                <div class="input-group">
                    <label for="bedrooms">🛏️ Bedrooms</label>
                    <input type="number" id="bedrooms" min="0" value="3" required>
                </div>

                <div class="input-group">
                    <label for="bathrooms">🛁 Bathrooms</label>
                    <input type="number" id="bathrooms" step="0.25" min="0" value="2" required>
                </div>

                <div class="input-group">
                    <label for="living_area">📐 Living Area (sqft)</label>
                    <input type="number" id="living_area" value="2000" required>
                </div>

                <div class="input-group">
                    <label for="lot_area">🌳 Lot Area (sqft)</label>
                    <input type="number" id="lot_area" value="5000" required>
                </div>

                <div class="input-group">
                    <label for="floors">🏢 Floors</label>
                    <input type="number" id="floors" step="0.5" min="1" value="1" required>
                </div>

                <div class="input-group">
                    <label for="waterfront">🌊 Waterfront Present</label>
                    <select id="waterfront">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="views">👁️ Number of Views</label>
                    <input type="number" id="views" min="0" max="4" value="0" required>
                </div>

                <div class="input-group">
                    <label for="condition">🏚️ House Condition (1-5)</label>
                    <input type="number" id="condition" min="1" max="5" value="3" required>
                </div>

                <div class="input-group">
                    <label for="grade">⭐ House Grade (1-13)</label>
                    <input type="number" id="grade" min="1" max="13" value="7" required>
                </div>

                <div class="input-group">
                    <label for="area_above">🏠 Area Excluding Basement (sqft)</label>
                    <input type="number" id="area_above" value="1500" required>
                </div>

                <div class="input-group">
                    <label for="area_basement">📦 Basement Area (sqft)</label>
                    <input type="number" id="area_basement" value="500" required>
                </div>

                <div class="input-group">
                    <label for="yr_built">📅 Built Year</label>
                    <input type="number" id="yr_built" min="1800" max="2026" value="1995" required>
                </div>

                <div class="input-group">
                    <label for="yr_renovated">🔨 Renovation Year (0 if none)</label>
                    <input type="number" id="yr_renovated" min="0" max="2026" value="0" required>
                </div>

                <div class="input-group">
                    <label for="lot_area_renov">🌾 Lot Area Renovated (sqft)</label>
                    <input type="number" id="lot_area_renov" value="5000" required>
                </div>

                <div class="input-group">
                    <label for="schools">🏫 Nearby Schools</label>
                    <input type="number" id="schools" min="0" value="2" required>
                </div>

                <div class="input-group">
                    <label for="dist_airport">✈️ Distance from Airport (km)</label>
                    <input type="number" id="dist_airport" step="0.1" value="15.0" required>
                </div>

            </div>

            <button type="submit" class="btn-submit" id="submitBtn">🚀 Calculate Price</button>
        </form>

        <div id="resultBox" class="result-box">
            <p>Predicted Price Value</p>
            <h2 id="resultOutput">-</h2>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const btn = document.getElementById('submitBtn');
            const resultBox = document.getElementById('resultBox');
            const resultOutput = document.getElementById('resultOutput');
            
            btn.innerHTML = '<span class="spinner"></span> Calculating...';
            btn.disabled = true;
            resultBox.style.display = 'none';

            // Feature vector payload matching exact order
            const payload = {
                features: [
                    parseFloat(document.getElementById('bedrooms').value),
                    parseFloat(document.getElementById('bathrooms').value),
                    parseFloat(document.getElementById('living_area').value),
                    parseFloat(document.getElementById('lot_area').value),
                    parseFloat(document.getElementById('floors').value),
                    parseInt(document.getElementById('waterfront').value),
                    parseFloat(document.getElementById('views').value),
                    parseFloat(document.getElementById('condition').value),
                    parseFloat(document.getElementById('grade').value),
                    parseFloat(document.getElementById('area_above').value),
                    parseFloat(document.getElementById('area_basement').value),
                    parseFloat(document.getElementById('yr_built').value),
                    parseFloat(document.getElementById('yr_renovated').value),
                    parseFloat(document.getElementById('lot_area_renov').value),
                    parseFloat(document.getElementById('schools').value),
                    parseFloat(document.getElementById('dist_airport').value)
                ]
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (response.ok) {
                    resultBox.className = 'result-box success';
                    // Format continuous prediction output as currency
                    const formattedValue = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(data.prediction);
                    resultOutput.innerText = formattedValue;
                } else {
                    resultBox.className = 'result-box error';
                    resultOutput.innerText = data.error || 'Prediction Failed';
                }
            } catch (err) {
                resultBox.className = 'result-box error';
                resultOutput.innerText = 'Network / Server Error';
            } finally {
                btn.innerHTML = '🚀 Calculate Price';
                btn.disabled = false;
                resultBox.style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pickle file is not available on server.'}), 500

    try:
        data = request.get_json(force=True)
        features = np.array([data['features']])
        
        # Run Linear Regression continuous prediction
        prediction = model.predict(features)[0]
        return jsonify({'prediction': float(prediction)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
