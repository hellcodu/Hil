import os
import json
import pickle
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
from config import Config
from models import db, User, Report
from prediction import predict_tumor
from gradcam import generate_gradcam
from ai_assistant import generate_ai_explanation
from pdf_report import generate_pdf_report

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'models'), exist_ok=True)

    with app.app_context():
        db.create_all()

    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/patient/ai-chat', methods=['POST'])
    def patient_ai_chat():
        if 'user_id' not in session or session.get('user_role') != 'patient':
            return {'success': False, 'error': 'Unauthorized'}, 401
            
        try:
            data = request.get_json(silent=True) or request.form
            question = data.get('question', '').strip() if data else ''
            report_id = data.get('report_id')
            
            if not question:
                return {'success': True, 'answer': 'Please enter a valid question.'}
                
            report_context = None
            if report_id:
                report = Report.query.get(report_id)
                if report and report.user_id == session['user_id']:
                    report_context = f"Prediction: {report.prediction}, Confidence: {report.confidence}%, Model: {report.model_name}"
                    
            answer = answer_patient_query(question, report_context)
            return {'success': True, 'answer': answer}
        except Exception as e:
            return {'success': True, 'answer': f"System error processing request: {str(e)}"}

  
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            role = request.form.get('role', 'patient')

            if User.query.filter_by(email=email).first():
                flash('Email address already registered.', 'error')
                return redirect(url_for('register'))

            new_user = User(name=name, email=email, role=role)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

        return render_template('register.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')

            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                session['user_id'] = user.id
                session['user_name'] = user.name
                session['user_role'] = user.role

                flash(f'Welcome back, {user.name}!', 'success')
                return redirect(url_for('doctor_dashboard') if user.role == 'doctor' else url_for('patient_dashboard'))
            else:
                flash('Invalid email or password.', 'error')
                return redirect(url_for('login'))

        return render_template('login.html')

    @app.route('/patient/dashboard')
    def patient_dashboard():
        if 'user_id' not in session or session.get('user_role') != 'patient':
            flash('Please log in as a patient.', 'error')
            return redirect(url_for('login'))
        reports = Report.query.filter_by(user_id=session['user_id']).order_by(Report.created_at.desc()).all()
        return render_template('patient_dashboard.html', reports=reports)

    @app.route('/patient/predict', methods=['POST'])
    def patient_predict():
        if 'user_id' not in session or session.get('user_role') != 'patient':
            return redirect(url_for('login'))

        if 'mri_image' not in request.files:
            flash('No image file selected.', 'error')
            return redirect(url_for('patient_dashboard'))

        file = request.files['mri_image']
        model_name = request.form.get('model_name', 'mobilenetv2')

        if file.filename == '':
            flash('No selected file.', 'error')
            return redirect(url_for('patient_dashboard'))

        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 1. Run Prediction
            pred_result = predict_tumor(file_path, model_name)
            if not pred_result['success']:
                flash(f"Prediction error: {pred_result.get('error')}", 'error')
                return redirect(url_for('patient_dashboard'))

            # 2. Generate Grad-CAM
            try:
                from prediction import load_prediction_model
                model_obj = load_prediction_model(model_name)
                gradcam_path = generate_gradcam(file_path, model_obj, model_name)
            except Exception as e:
                gradcam_path = None

            # 3. Generate AI Explanation
            ai_text = generate_ai_explanation(
                pred_result['prediction'],
                pred_result['confidence'],
                pred_result['probabilities'],
                model_name.upper()
            )

            # 4. Save Report to Database
            new_report = Report(
                user_id=session['user_id'],
                image_path=os.path.join('static', 'uploads', filename).replace('\\', '/'),
                prediction=pred_result['prediction'],
                confidence=pred_result['confidence'],
                probabilities=json.dumps(pred_result['probabilities']),
                model_name=model_name,
                gradcam_path=gradcam_path,
                ai_explanation=ai_text,
                review_status='Pending'
            )
            db.session.add(new_report)
            db.session.commit()

            flash('MRI analysis completed successfully!', 'success')
            return redirect(url_for('report_detail', report_id=new_report.id))

        flash('Invalid file type. Only JPG, JPEG, and PNG are allowed.', 'error')
        return redirect(url_for('patient_dashboard'))

    @app.route('/report/<int:report_id>')
    def report_detail(report_id):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        report = Report.query.get_or_404(report_id)
        return render_template('report_detail.html', report=report)

    @app.route('/doctor/dashboard')
    def doctor_dashboard():
        if 'user_id' not in session or session.get('user_role') != 'doctor':
            flash('Please log in as a doctor.', 'error')
            return redirect(url_for('login'))
        reports = Report.query.order_by(Report.created_at.desc()).all()
        return render_template('doctor_dashboard.html', reports=reports)

    @app.route('/doctor/review/<int:report_id>', methods=['GET', 'POST'])
    def doctor_review(report_id):
        if 'user_id' not in session or session.get('user_role') != 'doctor':
            flash('Please log in as a doctor.', 'error')
            return redirect(url_for('login'))

        report = Report.query.get_or_404(report_id)
        if request.method == 'POST':
            report.doctor_comment = request.form.get('doctor_comment')
            report.review_status = 'Reviewed'
            db.session.commit()
            flash('Clinical review submitted successfully!', 'success')
            return redirect(url_for('doctor_dashboard'))

        return render_template('doctor_review.html', report=report)

    @app.route('/model/performance')
    def model_performance():
        metrics_path = os.path.join('training_results', 'evaluation_metrics.pkl')
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
        return render_template('model_performance.html', metrics=metrics)

    @app.route('/download/pdf/<int:report_id>')
    def download_pdf(report_id):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        report = Report.query.get_or_404(report_id)
        
        report_data = {
            'report_id': report.id,
            'patient_name': report.patient.name,
            'model_name': report.model_name.upper(),
            'prediction': report.prediction,
            'confidence': report.confidence,
            'probabilities': json.loads(report.probabilities) if report.probabilities else {},
            'image_path': report.image_path,
            'gradcam_path': report.gradcam_path,
            'ai_explanation': report.ai_explanation,
            'doctor_comment': report.doctor_comment,
            'review_status': report.review_status
        }
        
        pdf_path = generate_pdf_report(report_data)
        return send_file(pdf_path, as_attachment=True)

    @app.route('/logout')
    def logout():
        session.clear()
        flash('You have been logged out.', 'success')
        return redirect(url_for('login'))

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)