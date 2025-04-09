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

        # Charts Database
        # c.execute('''
        #         CREATE TABLE IF NOT EXISTS Charts (
        #             id INT AUTO_INCREMENT PRIMARY KEY,
        #             name VARCHAR(255) NOT NULL,
        #             image VARBINARY(max) NOT NULL
        #             )
        #     ''')

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
                    response_time INT NOT NULL
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

 #returns true if successful, and false if not (email is identical to another user's)
 #all inputs are strings
 #owner account = "owner", customer account = "customer"
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
    createUser("topher@hotmail.com", "mykolyk", "owner")
    createUser("tberri50@stuy.edu", "instructorendorsed", "owner")
    createRestaurant("#GUDFAM Bagels", "8:00", "15:00", 20, "topher@hotmail.com")
    createRestaurant("Berri's Berry Smoothies", "7:00", "20:00", 30, "tberri50@stuy.edu")
    createTable("#GUDFAM Bagels", 10, 5, 7)
    createTable("#GUDFAM Bagels", 3, 3, 3)
    createTable("#GUDFAM Bagels", 5, 1, 1)
    createTable("Berri's Berry Smoothies", 1, 3, 7)
    createTable("Berri's Berry Smoothies", 2, 8, 5)
    createTable("Berri's Berry Smoothies", 4, 2, 6)
    createUser("marge@stuy.edu", "cslab", "customer")
    createReservation("marge@stuy.edu", 1, 2, "2025-6-27-11:10")
    createReservation("marge@stuy.edu", 1, 2, "2025-6-27-13:10")
    createReservation("marge@stuy.edu", 2, 3, "2025-6-27-14:10")
