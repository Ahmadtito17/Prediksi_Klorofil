from flask import Flask, render_template, request
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.history = []

def load_models(method):
    if method == 'regresi':
        model_A = joblib.load("../model_klorofil_a_linear.pkl")
        model_B = joblib.load("../model_klorofil_b_linear.pkl")
        model_total = joblib.load("../model_klorofil_total_linear.pkl")
    elif method == 'knn':
        model_A = joblib.load("../model_klorofil_a_knn.pkl")
        model_B = joblib.load("../model_klorofil_b_knn.pkl")
        model_total = joblib.load("../model_klorofil_total_knn.pkl")
    else:
        raise ValueError("Metode tidak dikenali")
    return model_A, model_B, model_total

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    method = request.form.get('method') or request.args.get('method') or 'regresi'
    pred_results, filename = [], None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename

            try:
                df = pd.read_excel(filepath)

                if 'Perlakuan' in df.columns and 'Pengambilan' in df.columns:
                    df_grouped = df.groupby(['Perlakuan', 'Pengambilan'])[['R', 'G', 'B']].mean().reset_index()
                else:
                    df_grouped = df.copy()

                df_grouped = df_grouped.fillna(0)
                df_grouped['Excess_Green'] = 2 * df_grouped['G'] - df_grouped['R'] - df_grouped['B']

                model_A, model_B, model_total = load_models(method)

                X = df_grouped[['Excess_Green']]
                df_grouped['Prediksi_Klorofil_A'] = model_A.predict(X)
                df_grouped['Prediksi_Klorofil_B'] = model_B.predict(X)
                df_grouped['Prediksi_Total_Klorofil'] = model_total.predict(X)

                pred_results = df_grouped.to_dict(orient='records')

                timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
                app.history.append({
                    'filename': filename,
                    'method': method,
                    'timestamp': timestamp,
                    'results': pred_results
                })

                # === VISUALISASI 2D ===
                def create_2d_plot(x, y, y_label, fname, color):
                    plt.figure(figsize=(8,6))
                    plt.scatter(x, y, c=color, edgecolor='k')
                    plt.xlabel('Excess Green')
                    plt.ylabel(y_label)
                    plt.title(f'Excess Green vs {y_label} ({method.upper()})')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
                    plt.close()

                create_2d_plot(df_grouped['Excess_Green'], df_grouped['Prediksi_Klorofil_A'],
                               'Prediksi Klorofil A', 'plot_a.png', 'red')
                create_2d_plot(df_grouped['Excess_Green'], df_grouped['Prediksi_Klorofil_B'],
                               'Prediksi Klorofil B', 'plot_b.png', 'green')
                create_2d_plot(df_grouped['Excess_Green'], df_grouped['Prediksi_Total_Klorofil'],
                               'Prediksi Total Klorofil', 'plot_total.png', 'blue')

            except Exception as e:
                pred_results = [{"error": f"Terjadi kesalahan: {str(e)}"}]

    return render_template(
        'index.html',
        results=pred_results,
        method=method,
        filename=filename
    )

@app.route('/history')
def history():
    return render_template('history.html', history=app.history)

if __name__ == '__main__':
    app.run(debug=True)
