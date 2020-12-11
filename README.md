# Practicum II Course Project
This is an overview of my practicum II course project

## The project is a continuation of my Practicum I project and focuses on implementation of the model. The project consist of an API built in Flask and provides several different functions.

First, lets go over what the application all includes:

![Image of Overall](https://github.com/sandsoftime660/PracticumII/blob/main/Images/Slide15.PNG)

### We will cover all of these in greater detail below:

Lets take a look at what and API is:

![Image of Overall](https://github.com/sandsoftime660/PracticumII/blob/main/Images/Slide11.PNG)

The application was built using Flask. This provided an excellent framework to build from

![Image of Overall](https://github.com/sandsoftime660/PracticumII/blob/main/Images/Slide13.PNG)

A background process (runs every 24 hours) was built to test the distributions of independent features. The goal here was to identify if our features were changing over time and if the model was in need of re-calibration

![Image of Overall](https://github.com/sandsoftime660/PracticumII/blob/main/Images/Slide16.PNG)

An email was generated to notify the analytics team of any distribution differences found

![Image of Overall](https://github.com/sandsoftime660/PracticumII/blob/main/Images/Slide17.PNG)

The application provides a scoring function which accepts and returns JSON. The function will do several different things from preprocessing the data to storing results in the database

![Image of Overall](https://github.com/sandsoftime660/PracticumII/blob/main/Images/Slide18.PNG)

Postman was used as a client to test the application

![Image of Overall](https://github.com/sandsoftime660/PracticumII/blob/main/Images/Slide19.PNG)

Finally, a dashboard app was created to visualize any feature drift. This is really the beginning for this part of the application. Future things may include model result comparisons or other metrics

![Image of Overall](https://github.com/sandsoftime660/PracticumII/blob/main/Images/Slide20.PNG)

# A video of this project with a demo is available here: 

# This project was a bit more difficult to present since the real work is all backend processes. There are several different notebooks, python scripts, and data files in this repository. Feel free to browse through them. They are really there for reference since several servers and databases are needed to run the code.  
