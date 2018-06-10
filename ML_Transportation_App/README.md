# Introduction
This project is to demonstrate using statistic analysis and machine learning to analyze data from a transportation app. The data comes from an interview project and thus is not fully disclosed.

# Project description
A company working with a transportation app is interested in predicting which driver signups are likely to start driving. To help explore this problem, they provided a dataset of a cohort of driver signups in January 2015. The data contains background information about the drivers and their car, as well as when they completed their first trip since the data was collected a few months later. The goal of this project is to 1) building a reliable model to predict whether or not the driver will start driving based on the given data, 2) identify key features responsible for the predicted results and 3) come up with practical suggestions to improve the completion rate of signed_up drivers. 

# Method
I will start with some exploratory statistical analysis to get some insights of the data, followed by using machine learning methods to predict whether each driver will complete their first trip, identify key responsible features and make practical suggestions to improve performance of the signed_up drivers.

All above anlysis will be done in Jupyter Notebook and Python 3.6.

# Data
Data is stored in the data.csv file and it contains the following columns:
- id: driver_id
- city_id: city_id this user signed up in
- signup_os: operating system the user used to sign up ('android', 'ios', 'website', 'other')
- signup_channel what channel did the driver sign up from ('offline', 'paid', 'organic', 'referral')
- signup_timestamp: time of the account creation; in the form 'YYYY-MM-DD'
- bgc_date: date of background check consent; in the form 'YYYY-MM-DD'
- vehicle_added_date: date when the driver's vehicle information was uploaded; in the form 'YYYY-MM-DD'
- first_trip_date: date of the first trip as the driver; in the form 'YYYY-MM-DD'
- vehicle_make: maker of the vehicle uploaded (i.e., Honda, Ford, Kia etc)
- vehicle_model: model of vehicle uploaded (i.e., Accord, Prius, Focus etc)
- vehicle_year: year that the car was made, in the form 'YYYY'
