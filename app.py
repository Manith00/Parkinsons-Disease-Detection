from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Flask Setup ===
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# === File Upload Config ===
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the .h5 model (Keras 3.x)
model_spiral = load_model('F1_spiral_mobilenetv2_finetuned.h5', compile=False)
model_wave   = load_model('F2_wave_mobilenetv2_finetuned.h5', compile=False)

# Export as SavedModel (folder)
model_spiral.export('F1_spiral_model')  
model_wave.export('F2_wave_model')# folder

# === Labels ===
class_names = ['Healthy', 'Parkinson']

# === Dummy Temporary Users ===
temp_users = {}

# === Routes ===

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm']

        if password != confirm:
            flash("Passwords do not match", "danger")
        elif username in temp_users:
            flash("User already exists", "warning")
        else:
            temp_users[username] = {'email': email, 'password': password}
            flash("Registration successful", "success")
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = temp_users.get(username)

        if user and user['password'] == password:
            session['username'] = username
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully", "info")
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash("Please login first", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        drawing_type = request.form['drawing_type']
        file = request.files['image']

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Select model
            if drawing_type == 'spiral':
                model = model_spiral
            else:
                model = model_wave

            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction) * 100)

            return render_template('result.html',
                                   predicted_label=predicted_class,
                                   confidence=confidence,
                                   drawing_type=drawing_type.title(),  # Capitalized for display
                                   uploaded_image=filename)


    return render_template('predict.html')

@app.route('/charts')
def charts():
    return render_template('chart.html')

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)

