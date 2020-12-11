# This API will score a claim for the fradu model. There are several steps involved and each will be commented as such. 

# https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

from flask import Flask, request, jsonify
import flask 
import json
import joblib
import pickle
import numpy as np
import dash  
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import database as db
from dashboard import instantiate_dash

server = Flask(__name__)

# Dash app creation:
instantiate_dash(server)


#This context root will score a claim 
@server.route("/score")
def score():
    print('Score function called from Postman')
    print('')
    # Converting to json 
    data = request.get_json()
    user = data['user']
    password = data['password']
    policy_num = data['policy_num']
    print('     Json input (without password displayed): ')
    print(              'User Calling Score Function: ' + user)
    print(              'Policy to Run Model on: ' + policy_num)
    # Finding the policy in database 
    print('')
    print('     Selecting Policy Data From Database...')
    claim = db.policy_pull(policy_num, user, password)

    # applying capping values... 

    # Removing columns not in training
    claim = claim[originalColumns]

    # Grabbing only the numeric columns
    numericCols = pd.DataFrame(claim.select_dtypes(exclude=['object'])).columns
    
    for var in numericCols:  
    
        feature = claim[var]

        # Calculate boundaries
        lower = capping.loc[capping.feature == var].lower.values[0]
        upper = capping.loc[capping.feature == var].upper.values[0]
        # Replace outliers
        claim[var] = np.where(feature > upper, upper, np.where(feature < lower, lower, feature))
    print('     Applying Capping Logic to Outliers...')
    # Loading in encoder...
    from sklearn import preprocessing
    import category_encoders as ce
    # le = preprocessing.LabelEncoder()

    # loading the encoder
    with open(path + '\encoder.pkl', 'rb') as file:
        le = pickle.load(file)

    # Here, we are dropping the target. Obviously, we wouldn't have a target in production :)
    claim.drop(['fraud_reported'], axis=1, inplace=True)

    claim = le.transform(claim)
    print('     Loading and Applying Categorical Encoders...')
    # Creating features...
    import featuretools as ft

    # Make an entityset and add the entity
    es = ft.EntitySet(id = 'claims')
    es.entity_from_dataframe(entity_id = 'data', dataframe = claim, 
                            make_index = True, index = 'index')

    # Run deep feature synthesis with transformation primitives
    feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'data',
                                        trans_primitives = ['add_numeric', 'multiply_numeric'])
    print('     Creating Interaction Features From Feature Tools...')
    # Predicting...
    prediction = grid.predict(feature_matrix)
    print('     Model Predicting...')
    print('')
    # Put to words
    if prediction == 1:
        predicted = 'Fraud'
        print('     Prediction: Fraud')
    else:
        predicted = 'Not Fraud'
        print('     Prediction: Not Fraud')

    # Find and create message as to why it was fraud 
    from eli5 import explain_prediction_df

    # grab cols from grid
    cols = grid.best_estimator_.named_steps['fs'].get_support(indices=True)
    feature_matrix = feature_matrix.iloc[:,cols]

    exp = explain_prediction_df(grid.best_estimator_.named_steps["clf"], feature_matrix.iloc[0])
    exp = exp.to_dict()
    print('')
    print('     Calling ELI5 for Prediction Explanation...')

    # Add to original json 
    del data['user']
    del data['password']
    data['predicted'] = predicted

    data['explain'] = exp
    print('     Storing the Prediction Score Results in Database...')
    #Store the results in the database
    db.store_score(data['policy_num'], prediction[0])
    print('     Json returned successfully! Below is an example of the return output: ')
    print('')
    print('')
    print(data)

    return data
    # return jsonify({'Prediction Results': str(type(data))})

# This context root will redirect the app to the dashboard 
@server.route("/dashboard")
def dashboard_render():
    return flask.redirect('/dash_dashboard/')

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    # This first part of the server will load the model and all objects needed to score a claim 
    print('Application is Being Launched...')
    print('')
    # First, we will load the model into memory
    path = r"C:\Users\sands\OneDrive\Desktop\II_MSDS_Data_Practicum\Data"
    print('Loading the Serialized Model...')
    print('')
    grid = joblib.load(path + r'\\model_final_FINAL_Gridsearch.mdl')
    print('Calling Database for Cutoff Capping Values...')
    print('')
    # Next, we will load the cut off values from the database. Our user only has select privilege on the claims database
    capping = db.cutoff_pull('fraudapi', 'password')
    print('Loading Training Data Columns...')
    print('')
    # Loading the column names from training set...
    with open(path + '\columns2.pkl', 'rb') as file:
        columns = pickle.load(file)

    # Bringing our original columns back
    with open(path + '\original_columns.pkl', 'rb') as file:
        originalColumns = pickle.load(file)

    # This second part of the server will create a job that runs once every day to compare the distributions of independent features 
    # https://stackoverflow.com/questions/21214270/how-to-schedule-a-function-to-run-every-hour-on-flask
    import time
    import atexit
    from apscheduler.schedulers.background import BackgroundScheduler

    # This function will compare distributions and send an email with any differences 
    def compare():
        print('')
        print('Background Scheduled Job For Comparison Will Now Execute...')
        # First, we will grab the data from the database. Since this is just a test, we will be manipulating the data such that our distributions are different
        # We will use our test data as the 'most recent' data 
        print('     Pulling Training Data and New Data...')
        train = db.compare_pull('fraudapi', 'password', 'train_dataset')
        test = db.compare_pull('fraudapi', 'password', 'test_dataset')

        # NOTE: There are 2 notebooks in the git repository that cover changing a distribution for testing.  
        print('')
        print('     Performing Mann-Whitney-U Test...')
        print('')
        from scipy.stats import mannwhitneyu
        different = []
        for col in train.columns:
            u, p = mannwhitneyu(list(train[col]), list(test[col]))
            printP = str(p)
            print('             Independent Feature Currently Testing: ' + col)
            print('             Test Statistic Value: ' + str(printP))
            if p <= 0.05:
                different.append(col)
                print('             Test Conclusion Distributions Are in Fact Statistically Significantly Different!')
            else:
                print('             Test Conclusion Distributions Are NOT Statistically Significantly Different')
            print('')

        # If differences are found, send an email!
        # https://realpython.com/python-send-email/#option-2-setting-up-a-local-smtp-server
        if different:
            print('')
            print('             At Least One Features Distribution Was Found To Be Statistically Different Than The Training Data')
            print('')
            print('     An Email Is Now Be Generated...')
            print('')
            import smtplib, ssl
            port = 465  # For SSL
            smtp_server = "smtp.gmail.com"
            password = "1Sexything!"
            receiver = "sandsoftime660@gmail.com"
            sender = "sandsoftime660@gmail.com"
            subject = "Work Comp Model Independent Feature Distribution Differences Found"
            
            body = """
            There were differences found in feature distributions. Here are the features that may be drifting:
            
            {}
            
            Please visit the dashboard to verify results @ http://127.0.0.1:5000/dashboard """.format(different)

            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = receiver
            msg.attach(MIMEText(body, 'plain'))

            text = msg.as_string()

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server1:
                server1.login(sender, password)
                server1.sendmail(sender, receiver, text)

            print('Email has been sent successfully! Below is an example of the message for demo:')
            print('')
            print(text)
            print('')
            print('')
            print('The Features With Differences Found Will Now Be Added To The Dashboard...')

            # This section will save a graph for dashboard. Basically, we are writing an html file everyday with any differences. 
            from datetime import date
            import plotly.express as px
            print('     Creating The File...')
            filename = '\\' + str(date.today().strftime("%m_%d_%Y"))
            wholePath = path + filename
            
            # This section will open the file, create a dataframe with columns to compare, create graph, and save to html file
            import plotly.express as px
            import pickle 
            import os

            try:
                os.mkdir(wholePath)
            except OSError:
                print ("Directory %s already exists!" % wholePath)
            else:
                print ("Successfully created the directory %s " % wholePath)
            print('')
            print('     Creating The Visualizations...')
            for col in different:
                compare = pd.DataFrame(columns=[col, 'dataset'])
                compare_2 = pd.DataFrame(columns=[col, 'dataset'])

            #     Adding the training data
                compare[col] = train[col]
                compare['dataset'] = 'Training'

            #     Adding the most recent data
                compare_2[col] = test[col]
                compare_2['dataset'] = 'Recent'

            #     Append the two together
                final = compare.append(compare_2)


                fig = px.histogram(final, x=col, color="dataset",
                            marginal="box", # or violin, rug
                            hover_data=final.columns,
                            title = 'Comparison for '+col)
                newfilename = wholePath + '\.' + col + '.pkl'
                
            #     with open(newfilename, 'w') as f:
                pickle.dump(fig, open(newfilename, 'wb'))
            print('')
            print('     Storing the Visualizations...')
            print('')
        print('BACKGROUND JOB NOW COMPLETE')
        print('')
        print('')
        return 


    # UNCOMMENT TO RUN BACKGROUND JOB!
    print('')
    print('Creating Background Scheduler And Scheduling Background Jobs...')
    print('')
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=compare, trigger="interval", seconds=220)
    scheduler.start()
    # Shut down the scheduler when exiting the server
    atexit.register(lambda: scheduler.shutdown())

    # Run the app
    server.run(debug=True, use_reloader=False)