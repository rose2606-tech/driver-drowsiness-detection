from flask import Flask, render_template, Response, request, redirect, session, url_for
from detector import generate_frames
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["driver_monitor_db"]
users_collection = db["users"]

# ---------- ROUTES ----------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']

    # Check if user exists
    if users_collection.find_one({'username': username}):
        return "User already exists! Please log in instead."

    # Add new user
    users_collection.insert_one({'username': username, 'password': password})
    session['user'] = username
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    user = users_collection.find_one({'username': username, 'password': password})
    if user:
        session['user'] = username
        return redirect(url_for('dashboard'))
    else:
        return "Invalid username or password!"

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template('dashboard.html', user=session['user'])

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- MAIN ----------
if __name__ == '__main__':
    app.run(debug=True)
