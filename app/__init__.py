"""
ZealousZebras: Anastasia Lee, Suhana Kumar, Dua Baig, Christopher Louie
SoftDev
P04: Cybersecurity Scoop
2025-04-01
"""

from flask import Flask, render_template, request, redirect, session, flash
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

def makePie(arr):
    set_v = set()
    for v in arr:
        set_v.add(v)
    list_v = list(set_v)
    print(list_v)
    count = {}
    for i in range(len(list_v)):
        count[list_v[i]] = 0
    for v in arr:
        count[v] = count[v]+1
    return count

def makeHistogram(arr, start, end, inc):
    count=[]
    for i in range(int((end-start)/inc)):
        count.append(0)
    for i in arr:
        count[int((i-start)/inc)] += 1
    return count

@app.route('/country', methods=['GET', 'POST'])
def country():
    print(makePie(db.returnCategory("country")))   
    return render_template('country.html', logged_in = ('username' in session))

@app.route('/year', methods=['GET', 'POST'])
def year():
    print(makePie(db.returnCategory("year")))   
    return render_template('year.html', logged_in = ('username' in session))

@app.route('/attack_type', methods=['GET', 'POST'])
def attack_type():
    print(makePie(db.returnCategory("attack_type")))   
    return render_template('attack_type.html', logged_in = ('username' in session))

@app.route('/industry', methods=['GET', 'POST'])
def industry():
    print(makePie(db.returnCategory("industry")))   
    return render_template('industry.html', logged_in = ('username' in session))

@app.route('/loss', methods=['GET', 'POST'])
def loss():
    print(makeHistogram(db.returnCategory("loss"), 0, 100, 10))   
    return render_template('loss.html', logged_in = ('username' in session))

@app.route('/affected_users', methods=['GET', 'POST'])
def affected_users():
    print(makeHistogram(db.returnCategory("affected_users"), 0, 1000000, 100000))   
    return render_template('affected_users.html', logged_in = ('username' in session))

@app.route('/attack_source', methods=['GET', 'POST'])
def attack_source():
    print(makePie(db.returnCategory("source")))   
    return render_template('attack_source.html', logged_in = ('username' in session))

@app.route('/vulnerability', methods=['GET', 'POST'])
def vulnerability():
    print(makePie(db.returnCategory("vulnerability")))   
    return render_template('vulnerability.html', logged_in = ('username' in session))

@app.route('/defense', methods=['GET', 'POST'])
def defense():
    print(makePie(db.returnCategory("defense")))   
    return render_template('defense.html', logged_in = ('username' in session))

@app.route('/resolution', methods=['GET', 'POST'])
def resolution():
    print(makeHistogram(db.returnCategory("resolution"), 0, 80, 8))   
    return render_template('resolution.html', logged_in = ('username' in session))

@app.route('/ai', methods=['GET', 'POST'])
def ai_page():
    from ai import rmse, mae
    accuracy = round(100 - (rmse / 24) * 100)
    return render_template(
        'ai.html', 
        error_hours=round(rmse, 2),
        accuracy=accuracy
    )

@app.route('/data', methods=['GET','POST'])
def data_page():
    search_query = request.args.get('search', '').lower()
    sort_key = request.args.get('sort', 'year')
    sort_order = request.args.get('order', 'asc') ## ascending/desc order

    rows = db.getFilteredData(search_query, sort_key, sort_order)
    return render_template('data.html', data=rows, search_query=search_query, sort_key=sort_key, sort_order=sort_order, logged_in = ('username' in session))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if ('username' in session):
        return render_template("profile.html", logged_in=True, username = session['username'])
    else:
        return redirect('/login')

if __name__ == "__main__":
    app.debug = True
    app.run()
