"""
ZealousZebras: Anastasia Lee, Suhana Kumar, Dua Baig, Christopher Louie
SoftDev
P04: Cybersecurity Scoop
2025-04-01
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import calendar, os
from datetime import datetime
import db

app = Flask(__name__)
app.secret_key = 'your_secret_key'
# HOME PAGE, SHOULD PROMPT REGISTER OR LOGIN
db.resetDB()
db.getData()


@app.route('/', methods=['GET', 'POST'])
def homeBase():
    if('username' in session):
        return render_template('home.html', logged_in = True)
    return render_template('home.html', logged_in = False)

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('username', None)
    session.pop('password', None)
    return redirect("/")

@app.route('/login', methods=['GET', 'POST'])
def login():
    print("")
    return render_template("login.html")

@app.route('/auth_login', methods=['GET', 'POST'])
def auth_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if db.checkLogin(username, password) == False:
            flash("Invalid login information", 'danger')  # Show error message
            return redirect('/login')
        session['username'] = username
        return redirect('/')
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template("register.html")

@app.route('/auth_register', methods=['GET', 'POST'])
def auth_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if db.createUser(username, password) == False:
            flash("Invalid: Account exists already", 'danger')  # Show error message
            return redirect('/register')
        session['username'] = username
        session['password'] = password
        return redirect('/')
    return render_template("register.html")

@app.route('/charts', methods=['GET', 'POST'])
def charts():
    arr = db.returnCategory("vulnerability")
    print("hello")
    print(arr)
    return render_template('charts.html')


@app.route('/data', methods=['GET','POST'])
def data_page():
    search_query = request.args.get('search', '').lower()
    sort_key = request.args.get('sort', 'year')
    sort_order = request.args.get('order', 'asc') ## ascending/desc order

    rows = db.getFilteredData(search_query, sort_key, sort_order)
    return render_template('data.html', data=rows, search_query=search_query, sort_key=sort_key, sort_order=sort_order)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if ('username' in session):
        return render_template("profile.html")
    else:
        return redirect('/login')

if __name__ == "__main__":
    app.debug = True
    app.run()
