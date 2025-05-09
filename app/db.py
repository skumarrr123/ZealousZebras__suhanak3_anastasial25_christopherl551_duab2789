"""
ZealousZebras: Anastasia Lee, Suhana Kumar, Dua Baig, Christopher Louie
SoftDev
P04: Cybersecurity Scoop
2025-04-01
"""

import sqlite3, os, csv

DATABASE_NAME = "DATABASE.db"

def createTables():
    if os.path.exists(DATABASE_NAME):
        print("Database already exists!!!\nWill not create tables")
    else:
        print("Creating tables...")
        db = sqlite3.connect(DATABASE_NAME)
        c = db.cursor()

        #User Info
        c.execute('''
                CREATE TABLE IF NOT EXISTS UserData (
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                    )
            ''')

        # Cybersecurity Info
        c.execute('''
                CREATE TABLE IF NOT EXISTS CyberData (
                    country TEXT NOT NULL,
                    year INT NOT NULL,
                    attack_type TEXT NOT NULL,
                    industry TEXT NOT NULL,
                    loss DECIMAL NOT NULL,
                    affected_users INT NOT NULL,
                    source TEXT NOT NULL,
                    vulnerability TEXT NOT NULL,
                    defense TEXT NOT NULL,
                    resolution INT NOT NULL
                    )
            ''')

        db.commit()
        db.close()

        print("Tables successfully created \n")
        return True

#just call this when resetting db, it calls createTables
#if not, call neither
def resetDB():
    if os.path.exists(DATABASE_NAME):
        os.remove(DATABASE_NAME)
        print("Resetting DB")
        return createTables()
    else:
        print("Cannot reset database as database does not exist")
        print("Creating database")
        return createTables()

def createUser(username, password):
    print(f"Adding user {username}")
    db = sqlite3.connect(DATABASE_NAME)
    c = db.cursor()

    try:
        c.execute('INSERT INTO UserData VALUES (?, ?)', (username, password))
        db.commit()
        db.close()
        print("Successfully added user")
        return True
    except Exception as e:
        print("Failed to add user (does the user already exist in the database?)")
        db.close()
        return False

#import data from csv
def getData():
    with open("cyberdata.csv", "r") as file:
        arr = list(csv.reader(file))[1:]
    db = sqlite3.connect(DATABASE_NAME)
    c = db.cursor()
    for a in arr:
        c.execute('INSERT INTO CyberData VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (a[0], int(a[1]), a[2], a[3], float(a[4]), int(a[5]), a[6], a[7], a[8], int(a[9])))
    db.commit()
    db.close()


def returnCategory(cat):
    db = sqlite3.connect(DATABASE_NAME)
    c = db.cursor()
    c.execute('SELECT ' + cat + ' FROM CyberData')
    arr = c.fetchall()
    db.commit()
    db.close()
    resp = []
    for row in arr:
        resp.append(row[0])
    return resp

#filtering data by both searth and sort
def getFilteredData(search_query='', sort_key='year', sort_order='asc'):
    valid_sk = {
        'year': 'year',
        'loss': 'loss',
        'affected_users': 'affected_users',
        'resolution': 'resolution'
    }
    sort_order = 'DESC' if sort_order == 'desc' else 'ASC'

    db = sqlite3.connect(DATABASE_NAME)
    c = db.cursor()

    query = f"SELECT * FROM CyberData" 
    params = []

    if search_query:
        query += """
        WHERE 
            lower(country) LIKE ? OR
            lower(attack_type) LIKE ? OR
            lower(industry) LIKE ? OR
            lower(source) LIKE ? OR
            lower(vulnerability) LIKE ? OR
            lower(defense) LIKE ?
        """
        like = f"%{search_query}%"
        params.extend([like] * 6)

    if sort_key in valid_sk:
        query += f" ORDER BY {valid_sk[sort_key]} {sort_order}"

    c.execute(query, params)
    rows = c.fetchall()
    db.close()
    return rows
    

def checkLogin(username, password):
    print(f"Checking login for {username}")
    db = sqlite3.connect(DATABASE_NAME)
    c = db.cursor()
    c.execute("SELECT password FROM UserData WHERE username = ?", (username,))
    row = c.fetchone()

    if row == None:
        print("Username does not exist in db")
        return False #account w that email does not exist

    if row[0] == password:
        print("Login correct")
        return True
    else:
        print("Incorrect password")
        return False

#will reset DB and add some data
def createSampleData():
    resetDB()
