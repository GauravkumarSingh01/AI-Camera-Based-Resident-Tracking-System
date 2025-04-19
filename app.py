from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import os
import csv
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import shutil
import threading
from datetime import timedelta
import json
import base64
import uuid

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'track_nest_secret_key')
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'TRAINING_FOLDER': 'TrainingImage',
    'STUDENT_DETAILS': 'StudentDetails',
    'TRACKING_DETAILS': 'Tracking_details',
    'UNKNOWN_FACES': 'UnknownFaces',
    'EMERGENCY_CAPTURES': 'EmergencyCaptures'
})

# Track recently detected residents with a cooldown period
RESIDENT_COOLDOWN_SECONDS = 120  # 2 minutes
resident_last_detected = {}  # Maps resident ID to last detection time 

# Add a context processor to make 'now' available to the all templates
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Ensure necessary directories exist
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

for folder in [app.config['UPLOAD_FOLDER'], app.config['TRAINING_FOLDER'], 
               app.config['STUDENT_DETAILS'], app.config['TRACKING_DETAILS'], 
               'TrainingImageLabel', app.config['UNKNOWN_FACES'], 
               app.config['EMERGENCY_CAPTURES']]:
    assure_path_exists(folder)

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        
        # Check if password file exists
        if os.path.isfile("TrainingImageLabel/psd.txt"):
            with open("TrainingImageLabel/psd.txt", "r") as f:
                stored_password = f.read().strip()
                if password == stored_password:
                    session['logged_in'] = True
                    return redirect(url_for('index'))
                else:
                    flash('Incorrect password!', 'danger')
        else:
            # First time setup
            with open("TrainingImageLabel/psd.txt", "w") as f:
                f.write(password)
            session['logged_in'] = True
            flash('Password set successfully!', 'success')
            return redirect(url_for('index'))
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Main routes
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Count total registrations
    registration_count = 0
    if os.path.isfile(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv"):
        with open(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", 'r') as csvFile:
            reader = csv.reader(csvFile)
            for _ in reader:
                registration_count += 1
        registration_count = (registration_count // 2) - 1 if registration_count > 0 else 0
    
    # Get today's tracking data
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    tracking_file = f"{app.config['TRACKING_DETAILS']}/Record_of_{today}.csv"
    tracking_data = []
    
    if os.path.isfile(tracking_file):
        df = pd.read_csv(tracking_file)
        for _, row in df.iterrows():
            tracking_data.append({
                'id': row['Id'] if 'Id' in row else '',
                'name': row['Name'] if 'Name' in row else '',
                'date': row['Date'] if 'Date' in row else '',
                'time': row['Time'] if 'Time' in row else ''
            })
    
    return render_template('index.html', 
                          registration_count=registration_count,
                          tracking_data=tracking_data,
                          today=today)

# Registration routes
@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Process registration form
        form_data = request.form
        
        # Basic validation
        required_fields = ['id', 'name', 'parent_name', 'phone', 'address']
        for field in required_fields:
            if not form_data.get(field):
                flash(f'Missing required field: {field}', 'danger')
                return redirect(url_for('registration'))
        
        # Store data in session for face capture
        session['registration_data'] = {
            'id': form_data.get('id'),
            'name': form_data.get('name'),
            'parent_name': form_data.get('parent_name'),
            'phone': form_data.get('phone'),
            'address': form_data.get('address'),
            'relationship': form_data.get('relationship', 'Parent'),
            'email': form_data.get('email', ''),
            'emergency_contact': form_data.get('emergency_contact', ''),
            'blood_group': form_data.get('blood_group', 'Not Specified'),
            'notes': form_data.get('notes', '')
        }
        
        # Redirect to face capture
        return redirect(url_for('face_capture'))
    
    return render_template('registration.html')

@app.route('/face-capture')
def face_capture():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if not session.get('registration_data'):
        flash('Please fill the registration form first', 'warning')
        return redirect(url_for('registration'))
    
    return render_template('face_capture.html')

@app.route('/process-face-capture', methods=['POST'])
def process_face_capture():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    if not session.get('registration_data'):
        return jsonify({'success': False, 'message': 'No registration data found'})
    
    try:
        # Get the captured image data
        image_data = request.json.get('image_data')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received'})
        
        # Remove header from base64 data
        image_data = image_data.split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        
        # Get registration data
        reg_data = session.get('registration_data')
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected in the image. Please try again with better lighting and positioning.'})
        
        if len(faces) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one person is in the frame.'})
        
        # Get serial number
        serial = 0
        if os.path.isfile(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv"):
            with open(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", 'r') as csvFile:
                reader = csv.reader(csvFile)
                for _ in reader:
                    serial += 1
            serial = (serial // 2)
        else:
            # Create CSV with header if it doesn't exist
            columns = ['SERIAL NO.', '', 'ID', '', 'NAME', '', 'PARENT NAME', '', 'PHONE', '', 
                      'ADDRESS', '', 'RELATIONSHIP', '', 'EMAIL', '', 'EMERGENCY CONTACT', '', 
                      'BLOOD GROUP', '', 'NOTES']
            with open(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(columns)
            serial = 1
        
        # Extract and save multiple sample images for better training
        for i in range(5):  # Take 5 samples with slight variations
            sample_num = i + 1
            for (x, y, w, h) in faces:
                # Add slight variation to the face region (if not the first sample)
                if i > 0:
                    variation = 3  # 3 pixels variation
                    x_var = max(0, x - variation if i % 2 == 0 else x + variation)
                    y_var = max(0, y - variation if i % 3 == 0 else y + variation)
                    # Ensure we don't go out of bounds
                    x_var = min(x_var, gray.shape[1] - w)
                    y_var = min(y_var, gray.shape[0] - h)
                else:
                    x_var, y_var = x, y
                
                # Save face image
                face_roi = gray[y_var:y_var+h, x_var:x_var+w]
                # Standardize the face size
                face_roi = cv2.resize(face_roi, (200, 200))
                # Apply histogram equalization for better light normalization
                face_roi = cv2.equalizeHist(face_roi)
                
                face_file = f"{app.config['TRAINING_FOLDER']}/{reg_data['name']}.{str(serial)}.{reg_data['id']}.{str(sample_num)}.jpg"
                cv2.imwrite(face_file, face_roi)
        
        # Save registration data to CSV
        row = [serial, '', reg_data['id'], '', reg_data['name'], '', reg_data['parent_name'], '', 
              reg_data['phone'], '', reg_data['address'], '', reg_data['relationship'], '', 
              reg_data['email'], '', reg_data['emergency_contact'], '', reg_data['blood_group'], '', 
              reg_data['notes']]
        
        with open(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        
        # Clear registration data from session
        session.pop('registration_data', None)
        
        # Return success
        return jsonify({
            'success': True, 
            'message': f'Registration completed for {reg_data["name"]} with ID {reg_data["id"]}. Face captured successfully with 5 variations for better recognition.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/train-model', methods=['GET'])
def train_model():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        # Check if there are any training images
        training_folder = app.config['TRAINING_FOLDER']
        if not os.path.exists(training_folder) or len(os.listdir(training_folder)) == 0:
            flash('No training images found. Please register someone first.', 'warning')
            return redirect(url_for('training_dashboard'))
            
        # Create a progress indicator for longer training time
        flash('Training started. This may take a moment...', 'info')
        
        # Train the model using the helper function
        result = train_face_recognition_model()
        
        if result['success']:
            flash(result['message'], 'success')
        else:
            flash(result['message'], 'danger')
        
    except Exception as e:
        flash(f'Error training model: {str(e)}', 'danger')
    
    return redirect(url_for('training_dashboard'))

def train_face_recognition_model():
    """
    Helper function to train the face recognition model.
    Returns a dictionary with success status and message.
    This can be called after registration or via the train-model route.
    """
    try:
        # Check if there are any training images
        training_folder = app.config['TRAINING_FOLDER']
        if not os.path.exists(training_folder) or len(os.listdir(training_folder)) == 0:
            return {
                'success': False,
                'message': 'No training images found. Please register someone first.'
            }
            
        # Initialize face recognizer with optimized parameters for NGO security
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,           # Use a small radius for detailed features
            neighbors=8,        # Consider 8 neighbors for better accuracy
            grid_x=8,           # Use an 8x8 grid for more precise recognition
            grid_y=8,
            threshold=80.0      # Lower threshold for better security at the NGO
        )
        
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        
        # Get training images and labels
        faces, IDs = get_images_and_labels(training_folder, detector)
        
        if len(faces) == 0:
            return {
                'success': False,
                'message': 'No faces detected in training images. Please register someone again with better images.'
            }
        
        # Train the model
        recognizer.train(faces, np.array(IDs))
        
        # Save the model
        recognizer.save("TrainingImageLabel/Trainner.yml")
        
        # Get training stats
        unique_ids = set(IDs)
        
        # Log training stats
        log_entry = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sample_count': len(faces),
            'unique_ids': len(unique_ids),
            'samples_per_person': round(len(faces) / len(unique_ids)) if len(unique_ids) > 0 else 0
        }
        
        # Create a more detailed message
        message = f'Model trained successfully with {len(faces)} face samples from {len(unique_ids)} residents!'
        message += f' (Average of {log_entry["samples_per_person"]} samples per person)'
        
        # Log training to a file
        training_log_file = "TrainingImageLabel/training_log.json"
        if os.path.exists(training_log_file):
            with open(training_log_file, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {'history': []}
        else:
            log_data = {'history': []}
        
        log_data['history'].append(log_entry)
        log_data['last_training'] = log_entry
        
        with open(training_log_file, 'w') as f:
            json.dump(log_data, f)
        
        return {
            'success': True,
            'message': message,
            'sample_count': len(faces),
            'unique_ids': len(unique_ids)
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error training model: {str(e)}'
        }

# Helper function to get images and labels for training
def get_images_and_labels(path, detector):
    """
    Function to get face images and corresponding labels for training.
    Improved to handle a larger number of samples efficiently.
    """
    # Get all image paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    face_samples = []
    ids = []
    processed_count = 0
    
    # Group images by ID to ensure balanced training
    id_to_images = {}
    for image_path in image_paths:
        try:
            # Get ID from the image filename (format: name.serial.id.sample.jpg)
            parts = os.path.basename(image_path).split('.')
            if len(parts) >= 4:
                id_num = int(parts[1])  # Get the serial number
                if id_num not in id_to_images:
                    id_to_images[id_num] = []
                id_to_images[id_num].append(image_path)
        except Exception as e:
            print(f"Error parsing {image_path}: {str(e)}")
    
    # Process images for each ID
    for id_num, images in id_to_images.items():
        # Use up to 100 images per ID, but keep at least 20 if available
        samples_per_id = min(len(images), 100)
        selected_images = images[:samples_per_id]
        
        for image_path in selected_images:
            try:
                # Load image and convert to grayscale
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Could not read image: {image_path}")
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # If this is a pre-processed image (200x200), use it directly
                if gray.shape[0] == 200 and gray.shape[1] == 200:
                    face_samples.append(gray)
                    ids.append(id_num)
                    processed_count += 1
                    continue
                
                # Detect faces in the grayscale image
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,  # Use a smaller scale factor for more detections
                    minNeighbors=4,   # Slightly more lenient to catch more faces
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # For each face detected, append to face samples and IDs
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    # Standardize size
                    face_roi = cv2.resize(face_roi, (200, 200))
                    # Apply histogram equalization for better light normalization
                    face_roi = cv2.equalizeHist(face_roi)
                    
                    face_samples.append(face_roi)
                    ids.append(id_num)
                    processed_count += 1
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    print(f"Processed {processed_count} face samples for training")
    return face_samples, ids

# Live tracking
@app.route('/live-tracking')
def live_tracking():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template('live_tracking.html')

@app.route('/api/get-tracking-stats')
def get_tracking_stats():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    # Get today's date
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    tracking_file = f"{app.config['TRACKING_DETAILS']}/Record_of_{today}.csv"
    
    authorized_count = 0
    unauthorized_count = 0
    
    if os.path.isfile(tracking_file):
        df = pd.read_csv(tracking_file)
        authorized_count = df[df['Id'] != 'Unknown'].shape[0]
        unauthorized_count = df[df['Id'] == 'Unknown'].shape[0]
    
    # Count unauthorized faces
    if os.path.exists(app.config['UNKNOWN_FACES']):
        unauthorized_count += len([f for f in os.listdir(app.config['UNKNOWN_FACES']) 
                                 if f.startswith('Unknown_Frame_') and 
                                 f.split('_')[2].startswith(datetime.datetime.now().strftime('%Y%m%d'))])
    
    return jsonify({
        'success': True,
        'authorized': authorized_count,
        'unauthorized': unauthorized_count,
        'total': authorized_count + unauthorized_count
    })

# Management routes
@app.route('/view-members')
def view_members():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    members = []
    if os.path.isfile(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv"):
        df = pd.read_csv(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv")
        for i in range(0, len(df), 2):
            if i+1 < len(df):
                row = df.iloc[i:i+2].values.flatten()
                member = {
                    'serial': row[0] if len(row) > 0 else '',
                    'id': row[2] if len(row) > 2 else '',
                    'name': row[4] if len(row) > 4 else '',
                    'parent_name': row[6] if len(row) > 6 else '',
                    'phone': row[8] if len(row) > 8 else '',
                    'address': row[10] if len(row) > 10 else '',
                    'relationship': row[12] if len(row) > 12 else '',
                    'email': row[14] if len(row) > 14 else '',
                    'emergency_contact': row[16] if len(row) > 16 else '',
                    'blood_group': row[18] if len(row) > 18 else '',
                    'notes': row[20] if len(row) > 20 else ''
                }
                members.append(member)
    
    return render_template('view_members.html', members=members)

@app.route('/delete-member/<member_id>', methods=['POST'])
def delete_member(member_id):
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    try:
        if os.path.isfile(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv"):
            df = pd.read_csv(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv")
            
            # Find and remove the member
            mask = df['ID'] != member_id
            if mask.all():
                return jsonify({'success': False, 'message': f'Member with ID {member_id} not found'})
            
            df_filtered = df[mask]
            df_filtered.to_csv(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", index=False)
            
            # Remove training images
            for file in os.listdir(app.config['TRAINING_FOLDER']):
                if f".{member_id}." in file:
                    os.remove(os.path.join(app.config['TRAINING_FOLDER'], file))
            
            return jsonify({'success': True, 'message': f'Member with ID {member_id} deleted successfully'})
        
        return jsonify({'success': False, 'message': 'No members found'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/emergency-alert', methods=['POST'])
def emergency_alert():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    try:
        # Save emergency details
        details = request.json.get('details', '')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a record of the emergency
        emergency_record = {
            'timestamp': timestamp,
            'details': details
        }
        
        with open(f"{app.config['EMERGENCY_CAPTURES']}/emergency_{timestamp}.json", 'w') as f:
            json.dump(emergency_record, f)
        
        return jsonify({'success': True, 'message': 'Emergency alert sent successfully!'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/change-password', methods=['GET', 'POST'])
def change_password():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate inputs
        if not all([old_password, new_password, confirm_password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('change_password'))
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'danger')
            return redirect(url_for('change_password'))
        
        # Check if old password is correct
        if os.path.isfile("TrainingImageLabel/psd.txt"):
            with open("TrainingImageLabel/psd.txt", "r") as f:
                stored_password = f.read().strip()
                if old_password != stored_password:
                    flash('Old password is incorrect', 'danger')
                    return redirect(url_for('change_password'))
                
            # Update password
            with open("TrainingImageLabel/psd.txt", "w") as f:
                f.write(new_password)
            
            flash('Password changed successfully', 'success')
            return redirect(url_for('index'))
        else:
            flash('No password set yet', 'danger')
            return redirect(url_for('login'))
    
    return render_template('change_password.html')

@app.route('/api/get-registration-data')
def get_registration_data():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    reg_data = session.get('registration_data')
    if not reg_data:
        return jsonify({'success': False, 'message': 'No registration data found'})
    
    return jsonify({
        'success': True,
        'id': reg_data.get('id', ''),
        'name': reg_data.get('name', '')
    })

@app.route('/api/recognize-face', methods=['POST'])
def recognize_face():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    try:
        # Clear expired detections first
        clear_expired_detections()
        
        # Check if model is trained
        if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
            return jsonify({
                'success': False, 
                'message': 'Face recognition model not trained. Please train the model first.'
            })
        
        # Get image data
        image_data = request.json.get('image_data')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received'})
        
        # Remove header from base64 data
        image_data = image_data.split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Initialize face recognizer with the same parameters used for training
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=100.0
        )
        recognizer.read("TrainingImageLabel/Trainner.yml")
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces - use slightly more sensitive parameters for live detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return jsonify({
                'success': False, 
                'message': 'No face detected in the image',
                'faces': []
            })
        
        # Load student details
        student_df = None
        if os.path.isfile(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv"):
            student_df = pd.read_csv(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv")
        
        today = datetime.datetime.now().strftime('%d-%m-%Y')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        current_datetime = datetime.datetime.now()
        
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            # Preprocess face for recognition
            face_roi = gray[y:y+h, x:x+w]
            # Standardize size - must match training size
            face_roi = cv2.resize(face_roi, (200, 200))
            # Apply histogram equalization for better light normalization
            face_roi = cv2.equalizeHist(face_roi)
            
            # Apply additional preprocessing - Gaussian blur to reduce noise
            face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
            
            # Predict
            id_pred, confidence = recognizer.predict(face_roi)
            
            # In LBPH, lower confidence means better match (0 is perfect match)
            # Convert to percentage where higher is better
            confidence_percentage = max(0, min(100, round(100 - confidence)))
            
            face_data = {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'confidence': confidence_percentage
            }
            
            # If confidence is high enough
            confidence_threshold = 40  # 40% confidence threshold
            recognition_successful = confidence_percentage >= confidence_threshold
            
            if recognition_successful:
                if student_df is not None:
                    # Extract ID from serial number
                    id_rows = student_df[student_df['SERIAL NO.'] == id_pred]
                    if not id_rows.empty:
                        # Get alternate rows which contain the actual data
                        idx = id_rows.index[0]
                        if idx + 1 < len(student_df):
                            # Construct row data spanning 2 rows
                            row_data = student_df.iloc[idx:idx+2].values.flatten()
                            
                            # Get the resident ID value
                            resident_id = row_data[2] if len(row_data) > 2 else str(id_pred)
                            
                            # Check if this resident is in cooldown period
                            is_in_cooldown = False
                            if resident_id in resident_last_detected:
                                time_diff = (current_datetime - resident_last_detected[resident_id]).total_seconds()
                                is_in_cooldown = time_diff < RESIDENT_COOLDOWN_SECONDS
                            
                            # Add recognition details to face data
                            face_data.update({
                                'id': resident_id,
                                'name': row_data[4] if len(row_data) > 4 else f"Unknown-{id_pred}",
                                'recognized': True,
                                'in_cooldown': is_in_cooldown,
                                'cooldown_seconds_remaining': int(RESIDENT_COOLDOWN_SECONDS - (current_datetime - resident_last_detected[resident_id]).total_seconds()) if is_in_cooldown else 0,
                                'details': {
                                    'serial': int(row_data[0]) if len(row_data) > 0 else 0,
                                    'parent_name': row_data[6] if len(row_data) > 6 else '',
                                    'relationship': row_data[12] if len(row_data) > 12 else '',
                                }
                            })
                            
                            # Record tracking only if not in cooldown
                            if not is_in_cooldown:
                                # Update the last detection time for this resident
                                resident_last_detected[resident_id] = current_datetime
                                
                                # Record the tracking details
                                tracking_file = f"{app.config['TRACKING_DETAILS']}/Record_of_{today}.csv"
                                
                                # Create tracking file if it doesn't exist
                                if not os.path.isfile(tracking_file):
                                    tracking_header = ['Id', 'Name', 'Date', 'Time', 'Confidence']
                                    with open(tracking_file, 'w', newline='') as f:
                                        writer = csv.writer(f)
                                        writer.writerow(tracking_header)
                                
                                # Add entry
                                tracking_row = [face_data['id'], face_data['name'], today, current_time, confidence_percentage]
                                with open(tracking_file, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(tracking_row)
                    else:
                        face_data.update({
                            'id': f"Unknown-{id_pred}",
                            'name': f"Unknown Person {id_pred}",
                            'recognized': False
                        })
                else:
                    face_data.update({
                        'id': f"Unknown-{id_pred}",
                        'name': f"Unknown Person {id_pred}",
                        'recognized': False
                    })
            else:
                # Unknown face, save it
                face_data.update({
                    'id': 'Unknown',
                    'name': 'Unknown Person',
                    'recognized': False
                })
                
                # Save unknown face with timestamp to avoid duplicates
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                unknown_file = f"{app.config['UNKNOWN_FACES']}/Unknown_Frame_{timestamp}_{str(uuid.uuid4())[:8]}.jpg"
                
                # Draw a red rectangle around the unknown face on the color image
                img_with_rectangle = img.copy()
                cv2.rectangle(img_with_rectangle, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Add text label above the face
                cv2.putText(img_with_rectangle, "Unknown Person", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save the complete color image instead of just the face
                cv2.imwrite(unknown_file, img_with_rectangle)
                
                # Record the unknown face in tracking
                tracking_file = f"{app.config['TRACKING_DETAILS']}/Record_of_{today}.csv"
                
                # Create tracking file if it doesn't exist
                if not os.path.isfile(tracking_file):
                    tracking_header = ['Id', 'Name', 'Date', 'Time', 'Confidence']
                    with open(tracking_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(tracking_header)
                
                # Add entry
                tracking_row = ['Unknown', 'Unknown Person', today, current_time, confidence_percentage]
                with open(tracking_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(tracking_row)
            
            recognized_faces.append(face_data)
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(faces)} faces',
            'faces': recognized_faces,
            'timestamp': current_time,
            'date': today
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/face-detection-check', methods=['POST'])
def face_detection_check():
    """
    Simple endpoint to check if a face is detected in an image.
    Used by the face capture page for real-time feedback.
    """
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    try:
        # Get image data
        image_data = request.json.get('image_data')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received'})
        
        # Remove header from base64 data
        image_data = image_data.split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_detected = len(faces) > 0
        face_location = None
        
        if face_detected and len(faces) == 1:
            x, y, w, h = faces[0]
            face_location = {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        
        # Return face detection result
        return jsonify({
            'success': True,
            'face_detected': face_detected,
            'face_count': len(faces),
            'face_location': face_location
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/capture-face-sample', methods=['POST'])
def capture_face_sample():
    """
    Captures a single face sample during the registration process.
    Part of capturing 100 samples for better face recognition.
    """
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    if not session.get('registration_data'):
        return jsonify({'success': False, 'message': 'No registration data found'})
    
    try:
        # Get the captured image data and sample number
        image_data = request.json.get('image_data')
        sample_num = request.json.get('sample_num', 1)
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received'})
        
        # Remove header from base64 data
        image_data = image_data.split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        
        # Get registration data
        reg_data = session.get('registration_data')
        
        # Initialize samples list in session if not already there
        if 'captured_samples' not in session:
            session['captured_samples'] = []
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return jsonify({
                'success': False, 
                'message': 'No face detected in the image'
            })
        
        if len(faces) > 1:
            return jsonify({
                'success': False, 
                'message': 'Multiple faces detected'
            })
        
        # Get serial number if we haven't stored it yet
        if 'serial_number' not in session:
            serial = 0
            if os.path.isfile(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv"):
                with open(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", 'r') as csvFile:
                    reader = csv.reader(csvFile)
                    for _ in reader:
                        serial += 1
                serial = (serial // 2)
            else:
                # Create CSV with header if it doesn't exist
                columns = ['SERIAL NO.', '', 'ID', '', 'NAME', '', 'PARENT NAME', '', 'PHONE', '', 
                          'ADDRESS', '', 'RELATIONSHIP', '', 'EMAIL', '', 'EMERGENCY CONTACT', '', 
                          'BLOOD GROUP', '', 'NOTES']
                with open(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", 'w', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(columns)
                serial = 1
            
            session['serial_number'] = serial
        else:
            serial = session['serial_number']
        
        # Extract face and save
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Standardize the face size
        face_roi = cv2.resize(face_roi, (200, 200))
        
        # Apply histogram equalization for better light normalization
        face_roi = cv2.equalizeHist(face_roi)
        
        # Create directory if it doesn't exist
        temp_dir = f"{app.config['UPLOAD_FOLDER']}/temp_{reg_data['id']}"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Save the face image to a temporary location
        temp_file = f"{temp_dir}/{reg_data['name']}.{str(serial)}.{reg_data['id']}.{str(sample_num)}.jpg"
        cv2.imwrite(temp_file, face_roi)
        
        # Add to session's captured samples list
        session['captured_samples'] = session.get('captured_samples', []) + [temp_file]
        # Make sure changes to the session are saved
        session.modified = True
        
        # Return success with face location
        return jsonify({
            'success': True,
            'message': f'Sample {sample_num} captured successfully',
            'face_location': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/complete-face-capture', methods=['POST'])
def complete_face_capture():
    """
    Completes the face capture process by moving temporary images to the training folder
    and saving the registration data.
    """
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    if not session.get('registration_data'):
        return jsonify({'success': False, 'message': 'No registration data found'})
    
    if not session.get('captured_samples'):
        return jsonify({'success': False, 'message': 'No face samples captured'})
    
    try:
        # Get registration data and captured samples
        reg_data = session.get('registration_data')
        captured_samples = session.get('captured_samples', [])
        serial = session.get('serial_number', 1)
        
        # Ensure we have enough samples
        if len(captured_samples) < 5:  # At least 5 samples needed
            return jsonify({
                'success': False, 
                'message': f'Not enough samples captured. Got {len(captured_samples)}, need at least 5.'
            })
        
        # Move the temporary files to the training folder
        for temp_file in captured_samples:
            if os.path.exists(temp_file):
                # Extract sample number from filename
                sample_num = os.path.basename(temp_file).split('.')[-2]
                
                # Create destination path
                dest_file = f"{app.config['TRAINING_FOLDER']}/{reg_data['name']}.{str(serial)}.{reg_data['id']}.{sample_num}.jpg"
                
                # Move/Copy the file
                shutil.copy2(temp_file, dest_file)
        
        # Save registration data to CSV
        row = [serial, '', reg_data['id'], '', reg_data['name'], '', reg_data['parent_name'], '', 
              reg_data['phone'], '', reg_data['address'], '', reg_data['relationship'], '', 
              reg_data['email'], '', reg_data['emergency_contact'], '', reg_data['blood_group'], '', 
              reg_data['notes']]
        
        with open(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        
        # Clean up temporary directory
        temp_dir = f"{app.config['UPLOAD_FOLDER']}/temp_{reg_data['id']}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Clear registration data and samples from session
        session.pop('registration_data', None)
        session.pop('captured_samples', None)
        session.pop('serial_number', None)
        
        # Automatically train the model with new data
        train_result = train_face_recognition_model()
        
        # Return success
        sample_count = len(captured_samples)
        message = f'Registration completed for {reg_data["name"]} with ID {reg_data["id"]}. {sample_count} face samples captured for improved recognition.'
        
        # Add training result to message if training was performed
        if train_result['success']:
            message += f' Model automatically updated with the new face data!'
        
        return jsonify({
            'success': True, 
            'message': message,
            'training_status': train_result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/index')
def index_redirect():
    return redirect(url_for('index'))

@app.route('/training-dashboard')
def training_dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Check if model is trained
    model_trained = os.path.isfile("TrainingImageLabel/Trainner.yml")
    
    # Get training statistics
    training_stats = {
        'total_faces': 0,
        'total_residents': 0,
        'last_training': None,
        'history': []
    }
    
    # Get training log if it exists
    training_log_file = "TrainingImageLabel/training_log.json"
    if os.path.exists(training_log_file):
        with open(training_log_file, 'r') as f:
            try:
                log_data = json.load(f)
                if 'last_training' in log_data:
                    training_stats['last_training'] = log_data['last_training']
                if 'history' in log_data:
                    training_stats['history'] = log_data['history'][-10:]  # Last 10 training sessions
            except json.JSONDecodeError:
                pass
    
    # Count registered residents
    registered_count = 0
    if os.path.isfile(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv"):
        with open(f"{app.config['STUDENT_DETAILS']}/StudentDetails.csv", 'r') as csvFile:
            reader = csv.reader(csvFile)
            for _ in reader:
                registered_count += 1
        registered_count = (registered_count // 2) - 1 if registered_count > 0 else 0
    
    training_stats['total_residents'] = registered_count
    
    # Count training images
    if os.path.exists(app.config['TRAINING_FOLDER']):
        image_count = len([f for f in os.listdir(app.config['TRAINING_FOLDER']) 
                          if os.path.isfile(os.path.join(app.config['TRAINING_FOLDER'], f))])
        training_stats['total_faces'] = image_count
    
    # Get system status
    system_status = {
        'model_trained': model_trained,
        'enough_data': training_stats['total_faces'] >= 20,  # At least 20 faces needed for good model
        'residents_registered': registered_count > 0
    }
    
    return render_template('training_dashboard.html', 
                          training_stats=training_stats,
                          system_status=system_status)

# Clear expired entries from resident detection tracking
def clear_expired_detections():
    """Remove entries older than the cooldown period from the resident tracking dict"""
    current_time = datetime.datetime.now()
    expired_ids = []
    
    for resident_id, detection_time in resident_last_detected.items():
        # Calculate time difference in seconds
        time_diff = (current_time - detection_time).total_seconds()
        if time_diff >= RESIDENT_COOLDOWN_SECONDS:
            expired_ids.append(resident_id)
    
    # Remove expired entries
    for expired_id in expired_ids:
        resident_last_detected.pop(expired_id, None)

@app.route('/api/get-recent-activity')
def get_recent_activity():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    # Get today's tracking data
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    tracking_file = f"{app.config['TRACKING_DETAILS']}/Record_of_{today}.csv"
    tracking_data = []
    
    if os.path.isfile(tracking_file):
        try:
            df = pd.read_csv(tracking_file)
            # Sort by time in descending order to get most recent first
            if 'Time' in df.columns:
                df = df.sort_values(by='Time', ascending=False)
            
            for _, row in df.iterrows():
                tracking_data.append({
                    'id': row['Id'] if 'Id' in row else '',
                    'name': row['Name'] if 'Name' in row else '',
                    'date': row['Date'] if 'Date' in row else '',
                    'time': row['Time'] if 'Time' in row else '',
                    'confidence': row['Confidence'] if 'Confidence' in row else ''
                })
        except Exception as e:
            print(f"Error reading tracking file: {str(e)}")
    
    return jsonify({
        'success': True,
        'tracking_data': tracking_data
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 