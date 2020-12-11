import pandas as pd
from sqlalchemy import create_engine

def policy_pull(policy_num, user, password):

    # Using an f string to input the user and password
    connstring = f'mysql+mysqlconnector://{user}:{password}@127.0.0.1:3306/claims'
    # Engine is a factory for connection. The connection does not happen here
    engine = create_engine(connstring, echo=False)
    # Connection happens here. Be sure to close
    dbConnection    = engine.connect()
    # Reading the table into a dataframe
    policy = pd.read_sql("select * from claims.test_dataset where policy_number = {}".format(policy_num), dbConnection);

    # Closing the connection
    dbConnection.close()

    return policy

def cutoff_pull(user, password):
    
    # Using an f string to input the user and password
    connstring = f'mysql+mysqlconnector://{user}:{password}@127.0.0.1:3306/claims'
    # Engine is a factory for connection. The connection does not happen here
    engine = create_engine(connstring, echo=False)
    # Connection happens here. Be sure to close
    dbConnection    = engine.connect()
    # Reading the table into a dataframe
    capping = pd.read_sql("select * from claims.cutoff_values", dbConnection);
    # Closing the connection
    dbConnection.close()

    return capping

def store_score(policy, prediction):

    # Using an f string to input the user and password
    connstring = f'mysql+mysqlconnector://fraudapi:password@127.0.0.1:3306/claims'
    # Engine is a factory for connection. The connection does not happen here
    engine = create_engine(connstring, echo=False)
    # Connection happens here. Be sure to close
    dbConnection = engine.connect()
    # Execute Query
    # capping = pd.read_sql("select * from claims.cutoff_values", dbConnection);

    # ins = claims_scored.insert()
    query = "insert into claims.claims_scored (claim, fraud) values ({}, {});".format(policy, prediction)
    dbConnection.execute(query)
    # dbConnection.execute('claims.claims_scored'.insert(), claim=data['policy_num'], fraud=data['predicted'],predicted_proba=data['proba'], json=data['explain'])
    # print(data['policy_num'])
    # Closing the connection
    dbConnection.close()

    return 

def compare_pull(user, password, table):

    # Using an f string to input the user and password
    connstring = f'mysql+mysqlconnector://{user}:{password}@127.0.0.1:3306/claims'
    # Engine is a factory for connection. The connection does not happen here
    engine = create_engine(connstring, echo=False)
    # Connection happens here. Be sure to close
    dbConnection    = engine.connect()
    # Reading the table into a dataframe
    data = pd.read_sql("select * from claims.{}".format(table), dbConnection);
    # Closing the connection
    dbConnection.close()

    return data