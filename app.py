from flask import Flask, request, render_template, redirect, url_for, flash
import pickle
import pandas as pd
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'fraud_detection_secret_key'  # Needed for flash messages

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('home'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Read the uploaded file
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                flash('Unsupported file format. Please upload CSV or Excel file.')
                return redirect(url_for('home'))
        except Exception as e:
            flash(f'Error reading file: {str(e)}')
            return redirect(url_for('home'))
        
        # Check if ID column exists, if not create one
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
            
        # Validate the 'type' column
        if 'type' not in df.columns:
            flash("File must contain a 'type' column")
            return redirect(url_for('home'))
            
        # Check if type values are valid (0-4)
        invalid_types = df[~df['type'].isin([0, 1, 2, 3, 4])]['type'].unique()
        if len(invalid_types) > 0:
            flash(f"Invalid values found in 'type' column. Only values 0-4 are allowed. Found: {', '.join(map(str, invalid_types))}")
            return redirect(url_for('home'))
        
        try:
            # Get required columns for prediction
            pred_columns = ['type', 'amount', 'payerdebited', 'recievercredited']
            
            # Check if all required columns exist
            missing_cols = [col for col in pred_columns if col not in df.columns]
            if missing_cols:
                flash(f"Missing required columns: {', '.join(missing_cols)}")
                return redirect(url_for('home'))
                
            # Create a copy of the dataframe with only the required columns for prediction
            df_pred = df[pred_columns].copy()
            
            # Make predictions
            predictions = model.predict(df_pred)
            probabilities = model.predict_proba(df_pred)
            
            # Add predictions and probabilities to the DataFrame
            df['prediction'] = ['Non-Fraudulent' if p == 0 else 'Fraudulent' for p in predictions]
            df['probability_non_fraud'] = [f"{prob[0] * 100:.2f}%" for prob in probabilities]
            df['probability_fraud'] = [f"{prob[1] * 100:.2f}%" for prob in probabilities]
            df['probability_fraud_raw'] = [prob[1] for prob in probabilities]  # Raw value for sorting
            
            # Calculate statistics
            fraud_count = len(df[df['prediction'] == 'Fraudulent'])
            non_fraud_count = len(df[df['prediction'] == 'Non-Fraudulent'])
            total_records = len(df)
            
            # Create separate dataframes for fraudulent and non-fraudulent
            fraudulent_df = df[df['prediction'] == 'Fraudulent'].sort_values('probability_fraud_raw', ascending=False)
            non_fraudulent_df = df[df['prediction'] == 'Non-Fraudulent']
            
            # Convert to records format for rendering
            fraud_records = fraudulent_df.to_dict('records')
            non_fraud_records = non_fraudulent_df.to_dict('records')
            
            # Prepare statistics summary
            stats = {
                'total_records': total_records,
                'fraud_count': fraud_count,
                'non_fraud_count': non_fraud_count,
                'fraud_percentage': f"{(fraud_count / total_records) * 100:.2f}%" if total_records > 0 else "0%"
            }
            
            # Render the template with all results
            return render_template('index.html', 
                                  fraud_results=fraud_records,
                                  non_fraud_results=non_fraud_records,
                                  stats=stats)
            
        except Exception as e:
            flash(f'Error processing data: {str(e)}')
            return redirect(url_for('home'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
