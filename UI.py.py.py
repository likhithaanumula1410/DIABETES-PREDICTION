#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing libraries
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Function for diabetes prediction
def predict_diabetes(data):
    # Importing dataset
    dataset = pd.read_csv('diabetes.csv')

    # separating the data and labels
    X = dataset.drop(columns = 'Outcome', axis=1)
    Y = dataset['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = dataset['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')

    #training the support vector Machine Classifier
    classifier.fit(X_train, Y_train)

    # accuracy score on the training data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    # accuracy score on the test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'Oops!.. The person is diabetic'
    
# Function to perform diabetes prediction
def predict():
    # Retrieve user input from the form
    glucose = float(Glucose_entry.get())
    blood_pressure = float(BloodPressure_entry.get())
    bmi = float(BMI_entry.get())
    pregnancies = float(Pregnancies_entry.get())
    skinThickness = float(SkinThickness_entry.get())
    insulin = float(Insulin_entry.get())
    diabetesPedigreeFunction = float(DiabetesPedigreeFunction_entry.get())
    age = float(Age_entry.get())

    

    # Perform some basic input validation (you can add more)
    if glucose < 0 or blood_pressure < 0 or bmi < 0 or pregnancies < 0 or skinThickness < 0 or insulin < 0 or diabetesPedigreeFunction < 0 or age < 0:
        messagebox.showerror("Error", "Please enter valid values.")
        return

    # Data for plotting
    x = ['Glucose', 'Blood Pressure', 'BMI','Pregnancies','skinThickness','insulin','diabetesPedigreeFunction','age']
    y = [glucose, blood_pressure, bmi,pregnancies,skinThickness,insulin,diabetesPedigreeFunction,age]

    # Create a simple bar plot
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_ylabel('Values')

    canvas = FigureCanvasTkAgg(fig, master=tab3)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0)

    # Perform diabetes prediction and display the result
    result = predict_diabetes([glucose, blood_pressure, bmi, pregnancies, skinThickness, insulin, diabetesPedigreeFunction, age])
    result_label.config(text=f"Diabetes Prediction: {result}")


# Function to enable prediction tab and disable login tab
def enable_prediction():
    tab_control.tab(tab2, state="normal")  # Enable prediction form
    tab_control.tab(tab3, state="normal")  # Enable result tab
    tab_control.tab(tab1, state="disabled")  # Disable login tab

# Function to reset the form and result
def reset_form():
    Glucose_entry.delete(0, 'end')
    BloodPressure_entry.delete(0, 'end')
    BMI_entry.delete(0, 'end')
    Pregnancies_entry.delete(0, 'end')
    SkinThickness_entry.delete(0, 'end')
    Insulin_entry.delete(0, 'end')
    DiabetesPedigreeFunction_entry.delete(0, 'end')
    Age_entry.delete(0, 'end')
    result_label.config(text="")

# Function to perform logout
def logout():
    reset_form()  # Reset the form and result
    tab_control.tab(tab1, state="normal")  # Enable login tab
    tab_control.tab(tab2, state="disabled")  # Disable prediction form
    tab_control.tab(tab3, state="disabled")  # Disable result tab

# Function to perform login
def login():
    username = username_entry.get()
    password = password_entry.get()
    # Replace this with your actual login validation
    if username == "vinni" and password == "vinni":
        enable_prediction()
        tab_control.select(tab2)  # Navigate to the prediction tab
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Create the main window
root = tk.Tk()
root.title("ML-Diabetes Prediction App")

# Create tabs for the application
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)
tab5 = ttk.Frame(tab_control)

tab_control.add(tab1, text="Login")
tab_control.add(tab2, text="Diabetes Prediction Form")
tab_control.add(tab3, text="Plot and Result")
tab_control.pack(expand=1, fill="both")
tab_control.add(tab4, text="Health Tips")
tab_control.add(tab5, text="About")


# Create a login screen
username_label = tk.Label(tab1, text="Username:")
password_label = tk.Label(tab1, text="Password:")
username_entry = tk.Entry(tab1)
password_entry = tk.Entry(tab1, show="*")
login_button = tk.Button(tab1, text="Login", command=login)

username_label.grid(row=0, column=0, padx=20, pady=20)
password_label.grid(row=1, column=0, padx=20, pady=20)
username_entry.grid(row=0, column=1, padx=20, pady=20)
password_entry.grid(row=1, column=1, padx=20, pady=20)
login_button.grid(row=2, column=0, columnspan=2, padx=20, pady=20)

# Create a form for diabetes prediction
Pregnancies_label = tk.Label(tab2, text="Pregnancies Level :")
Glucose_label = tk.Label(tab2, text="Glucose :")
BloodPressure_label = tk.Label(tab2, text="BloodPressure :")
SkinThickness_label = tk.Label(tab2, text="SkinThickness :")
Insulin_label = tk.Label(tab2, text="Insulin :")
BMI_label = tk.Label(tab2, text="BMI :")
DiabetesPedigreeFunction_label = tk.Label(tab2, text="DiabetesPedigreeFunction :")
Age_label = tk.Label(tab2, text="Age :")



Pregnancies_entry = tk.Entry(tab2)
Glucose_entry = tk.Entry(tab2)
BloodPressure_entry = tk.Entry(tab2)
SkinThickness_entry = tk.Entry(tab2)
Insulin_entry = tk.Entry(tab2)
BMI_entry = tk.Entry(tab2)
DiabetesPedigreeFunction_entry = tk.Entry(tab2)
Age_entry = tk.Entry(tab2)

predict_button = tk.Button(tab2, text="Predict Diabetes", command=predict)

Pregnancies_label.grid(row=0, column=0)
Glucose_label.grid(row=1, column=0)
BloodPressure_label.grid(row=2, column=0)
SkinThickness_label.grid(row=3, column=0)
Insulin_label.grid(row=4, column=0)
BMI_label.grid(row=5, column=0)
DiabetesPedigreeFunction_label.grid(row=6, column=0)
Age_label.grid(row=7, column=0)

Pregnancies_entry.grid(row=0, column=1)
Glucose_entry.grid(row=1, column=1)
BloodPressure_entry.grid(row=2, column=1)
SkinThickness_entry.grid(row=3, column=1)
Insulin_entry.grid(row=4, column=1)
BMI_entry.grid(row=5, column=1)
DiabetesPedigreeFunction_entry.grid(row=6, column=1)
Age_entry.grid(row=7, column=1)

predict_button.grid(row=9, column=0, columnspan=2)


# Create a label for the prediction result
result_label = tk.Label(tab2, text="", fg="white", bg="blue")
result_label.grid(row=len(labels)+2, column=0, columnspan=2)

# Add educational content to the Health Tips tab
# For example, you can add labels with text content or embed videos or images
tip_label = tk.Label(tab4, text="Here are some health tips for diabetes prevention and management:")
tip_label.pack()

# Example text content
tip_text = """
1. Maintain a healthy diet: Focus on eating plenty of fruits, vegetables, whole grains, and lean proteins. Limit your intake of sugary and processed foods.

2. Stay physically active: Aim for at least 30 minutes of moderate-intensity exercise most days of the week. Choose activities you enjoy, such as walking, swimming, or cycling.

3. Monitor your blood sugar levels: Regularly check your blood sugar levels as recommended by your healthcare provider. Keep track of your readings and discuss them with your doctor.

4. Take medications as prescribed: If you have been prescribed medication for diabetes, make sure to take it exactly as directed by your healthcare provider. 

5. Manage stress: Practice stress-reduction techniques such as deep breathing, meditation, or yoga to help manage your stress levels.

6. Get regular check-ups: Visit your healthcare provider regularly for check-ups and screenings. This can help identify any potential health issues early and prevent complications.

7. Stay informed: Educate yourself about diabetes and its management. Stay up-to-date on the latest research, guidelines, and treatment options.

Remember, small lifestyle changes can make a big difference in managing diabetes and improving your overall health.
"""
tip_text_label = tk.Label(tab4, text=tip_text, justify="left")
tip_text_label.pack(pady=10)
tip_text_label.config(font=("Arial", 8))

# Add content to the About tab
about_text = """
This application helps in predicting diabetes using machine learning algorithms.
It uses the SVM (Support Vector Machine) algorithm to make predictions based on input data.
Developed by 
1. [Vineesha Talari] - [VineeshaTalari@my.unt.edu]
2. [Lokesh Reddy Bommireddy] - [lokeshreddybommireddy@my.unt.edu]
3. [Sai Likhitha Anumula] - [sailikhithaanumula@my.unt.edu]
4. [Sai Sricharan Ayilavarapu] - [saisricharanayilavarapu@my.unt.edu]
5. [Viveksen Naroju] - [viveksennaroju@my.unt.edu]
"""
about_label = tk.Label(tab5, text=about_text, wraplength=500, justify='left')
about_label.pack(padx=20, pady=20)

# Create a logout button in the prediction tab
logout_button = tk.Button(tab2, text="Logout", command=logout)
logout_button.grid(row=0, column=4, columnspan=2)

# Disable prediction form and result tab initially
tab_control.tab(tab2, state="disabled")
tab_control.tab(tab3, state="disabled")

# Start the Tkinter main loop
root.mainloop()


# In[ ]:





# In[ ]:




