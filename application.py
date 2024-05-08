import os
from flask import Flask, render_template
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import re
import io
import zlib
from werkzeug.utils import secure_filename
from flask import Response
from cs50 import SQL
from flask import Flask, flash, jsonify, redirect, render_template, request, session ,url_for
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import face_recognition
from PIL import Image
from base64 import b64encode, b64decode
import re
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template
import joblib
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


from helpers import apology, login_required
# Configure application
app = Flask(__name__)
#configure flask-socketio

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Custom filter


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///data.db")
# Créer la table users si elle n'existe pas déjà
db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, password TEXT NOT NULL)")




@app.route("/")
@login_required
def home():


    return redirect("/home")

@app.route("/home")
@login_required
def index():
    input_username = request.form.get("username")
    current_user = db.execute("SELECT * FROM users WHERE username = :username",
                            username=input_username)
    return render_template("index.html")







@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")

        # Ensure username was submitted
        if not input_username:
            return render_template("login.html",messager = 1)



        # Ensure password was submitted
        elif not input_password:
             return render_template("login.html",messager = 2)

        # Query database for username
        username = db.execute("SELECT * FROM users WHERE username = :username",
                              username=input_username)

        # Ensure username exists and password is correct
        if len(username) != 1 or not check_password_hash(username[0]["hash"], input_password):
            return render_template("login.html",messager = 3)

        # Remember which user has logged in
        session["user_id"] = username[0]["id"]
        session["username"] = username[0]["username"]
        



        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")



@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")
        input_confirmation = request.form.get("confirmation")

        # Ensure username was submitted
        if not input_username:
            return render_template("register.html",messager = 1)

        # Ensure password was submitted
        elif not input_password:
            return render_template("register.html",messager = 2)

        # Ensure passwsord confirmation was submitted
        elif not input_confirmation:
            return render_template("register.html",messager = 4)

        elif not input_password == input_confirmation:
            return render_template("register.html",messager = 3)

        # Query database for username
        username = db.execute("SELECT username FROM users WHERE username = :username",
                              username=input_username)

        # Ensure username is not already taken
        if len(username) == 1:
            return render_template("register.html",messager = 5)

        # Query database to insert new user
        else:
            new_user = db.execute("INSERT INTO users (username, hash) VALUES (:username, :password)",
                                  username=input_username,
                                  password=generate_password_hash(input_password, method="pbkdf2:sha256", salt_length=8),)

            if new_user:
                # Keep newly registered user logged in
                session["user_id"] = new_user

            # Flash info for the user
            flash(f"Registered as {input_username}")

            # Redirect user to homepage
            return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")




@app.route("/facereg", methods=["GET", "POST"])
def facereg():
    session.clear()
    if request.method == "POST":


        encoded_image = (request.form.get("pic")+"==").encode('utf-8')
        username = request.form.get("name")
        name = db.execute("SELECT * FROM users WHERE username = :username",
                        username=username)
              
        if len(name) != 1:
            return render_template("camera.html",message = 1)

        id_ = name[0]['id']    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/unknown/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        try:
            image_of_bill = face_recognition.load_image_file(
            './static/face/'+str(id_)+'.jpg')
        except:
            return render_template("camera.html",message = 5)

        bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

        unknown_image = face_recognition.load_image_file(
        './static/face/unknown/'+str(id_)+'.jpg')
        try:
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        except:
            return render_template("camera.html",message = 2)


#  o    mpare faces
        results = face_recognition.compare_faces(
        [bill_face_encoding], unknown_face_encoding)

        if results[0]:
            username = db.execute("SELECT * FROM users WHERE username = :username",
                              username="swa")
            session["user_id"] = username[0]["id"]
            return redirect("/")
        else:
            return render_template("camera.html",message=3)


    else:
        return render_template("camera.html")



@app.route("/facesetup", methods=["GET", "POST"])
def facesetup():
    if request.method == "POST":


        encoded_image = (request.form.get("pic")+"==").encode('utf-8')


        id_=db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]
        # id_ = db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        image_of_bill = face_recognition.load_image_file(
        './static/face/'+str(id_)+'.jpg')    
        try:
            bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]
        except:    
            return render_template("face.html",message = 1)
        return redirect("/home")

    else:
        return render_template("face.html")

def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return render_template("error.html",e = e)





@app.route('/form', methods=['GET'])
def show_form():
    return render_template('prediction_form.html')

@app.route('/predict',methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])

    # Charger les données
    df = pd.read_excel('data/C__a_inal_df_preprocessed_2021111201.xlsx')

    # Prétraitement des données - supprimer les valeurs manquantes
    df = df.dropna(subset=['Temperature', 'Lost_quantity'])

    # Séparer les caractéristiques (X) et la cible (y)
    X = df[['Temperature']]
    y = df['Lost_quantity']

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Faire la prédiction
    predicted_lost_quantity = model.predict([[temperature]])

    # Renvoyer les résultats à afficher
    return render_template('result.html', predicted_lost_quantity=predicted_lost_quantity[0])







@app.route('/stockage')
def stockage():
  
  #Load inventory data
    inventory_data = pd.read_csv('/data/ORDER_RECIPE1.csv')

    # Convert 'START_TIME' to datetime
    inventory_data['START_TIME'] = pd.to_datetime(inventory_data['START_TIME'])

    # Dictionary to store MSE for each product group
    mse_per_product = {}

    # Define the product groups
    product_groups = ['group1', 'group2', 'group3']  # Update with your actual product groups

    # Loop over each product group
    for product_group in product_groups:
        # Filter data for the current product group
        product_data = inventory_data[inventory_data['PRODUCT_GROUP_CODE'] == product_group]

        print(f'Product group: {product_group}, Number of samples: {len(product_data)}')

        # Check if the dataset is empty
        if len(product_data) == 0:
            print(f'Skipping {product_group} due to empty dataset')
            continue

        # Group data by start time and sum product quantities
        grouped_data = product_data.groupby('START_TIME')['ORDERED_QUANTITY'].sum().reset_index()

        # Convert 'START_TIME' to timestamp
        grouped_data['START_TIME_TIMESTAMP'] = grouped_data['START_TIME'].apply(lambda x: x.timestamp())

        # Split data into features (X) and target variable (y)
        X = grouped_data[['START_TIME_TIMESTAMP']]  # Features: Start time
        y = grouped_data['ORDERED_QUANTITY']  # Target variable: Ordered quantity

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f'X_train size: {len(X_train)}, X_test size: {len(X_test)}, y_train size: {len(y_train)}, y_test size: {len(y_test)}')

        # Initialize and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred_test = model.predict(X_test)

        # Make predictions on the training set (for validation)
        y_pred_train = model.predict(X_train)

        # Calculate mean squared error for test and validation sets
        mse_test = mean_squared_error(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)

        # Store MSE for the current product group
        mse_per_product[product_group] = {'test': mse_test, 'train': mse_train}

    # Render the results template with MSE values
    return render_template('results.html', mse_per_product=mse_per_product)
# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

if __name__ == '__main__':
      app.run()
