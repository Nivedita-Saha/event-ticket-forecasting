import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
import matplotlib.pyplot as plt

# List of dataset filenames
latest_event = 'SRM23_test_encoded.csv' # The file to be used for predicting the sales
AllData_sets = [
    'D19_encoded.csv', 'D21_encoded.csv', 'GP21_encoded.csv',
    'MSE21_encoded.csv', 'NP21_encoded.csv', 'SRM22_encoded.csv', latest_event
]
IT_event = ['D19_encoded.csv', 'D21_encoded.csv', latest_event]
Property_event = ['GP21_encoded.csv', 'NP21_encoded.csv', latest_event]
Education_event = ['SRM22_encoded.csv', latest_event]

# List of different types of events
Event_list = {1: "IT Managers", 2: "Property Managers", 3: "Education Managers", 4: "All the Events"}

# Diplaying the different types of events, for Training Purpose
def display_menu():
    print("Please choose an event:")
    for key, value in Event_list.items():
        print(f"{key}: {value}")

# Selecting the type of datasets to be used
def event_name(name):
    if name == 1:
        selected_files = IT_event
        print('IT Managers dataset uploaded')
    elif name == 2:
        selected_files = Property_event
        print('Property Managers dataset uploaded')
    elif name == 3:
        selected_files = Education_event
        print('Education Managers dataset uploaded')
    elif name == 4:
        selected_files = AllData_sets
        print('All dataset uploaded')
    else:
        print('Wrong input, try again')
        return None
    
    dataframes = [pd.read_csv(file) for file in selected_files]
    df = pd.concat(dataframes, axis=0)
    dataframes = [pd.read_csv(file) for file in selected_files]
    df = pd.concat(dataframes, axis=0)
    df =  df.rename(columns={'Created Date': 'ds', 'Attendance Count': 'y'})
    print (df.reset_index(drop=True))
    return df 

# Calculation of how many days till the event day does the program need to forecast
def period(df, stop_date):
    last_date = df['ds'].max() # In a real world situation, this will be the last day tickets were sold
    period = (stop_date - last_date).days
    return period

# Actual prediction
def prediction(df, Start_date, stop_date):
    m = Prophet()
    m.fit(df)
    prediction_period = period(df, stop_date)
    future = m.make_future_dataframe(periods=prediction_period)
    future = future[future['ds'] > df['ds'].max()]
    forecast = m.predict(future)
    #forecast['yhat'] = forecast['yhat'].abs() # Incase we want to use absolute numbers only
    result = forecast[['ds', 'yhat']].tail()
    forecast = forecast[forecast['yhat']>0] 
    Predicted_results = forecast ['yhat'].sum()
    result['ds'] = pd.to_datetime(result['ds'])
    TrueDate_Ranges = df[df['ds'] >= Start_date]
    sold_todate = TrueDate_Ranges['y'].sum()
    sum_y_actual = sold_todate  + Predicted_results
    print('\n*************************************************')
    print(f'\nThe number of tickets already sold  is:\n{sold_todate }')
    print(f'From today till the event day, the number of tickets that may be sold given the current trajectory is:\n{round(Predicted_results)}')
    print(f'The Total number of tickets to be sold given the current trajectory is:\n{round(sum_y_actual)}')
    print('\n*************************************************')
    return forecast, m

# Making a scatter plot
def plot1(forecast, df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['ds'], df['y'], color='black', label='Actual')
    plt.scatter(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')
    plt.title('Predicted Ticket Sales')
    plt.xlabel('Date')
    plt.ylabel('Number of Tickets Sold')
    plt.legend()
    plt.show()

#Making line plot
def plot2(forecast, m):
    fig1 = m.plot(forecast)
    ax = fig1.gca()
    ax.set_title('Predicted Ticket Sales')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Tickets Sold')
    plt.show()
    return fig1
    
def main():

    while True:
        print("\nGroup 3 Prediction App")
        print("1. Display list of Events Present")
        print("2. Do a Prediction")
        print("3. Plot")
        print("0. Quit")

        selection = input("Welcome to the Prediction App, how can we assist you today? ")
        if selection == '1':
            print('*************************************************')
            display_menu()
            try:
                choice = int(input("Enter the number corresponding to the event: "))
                event_name(choice)
            except ValueError:
                print("Invalid input, please enter a number.")
            print('*************************************************')
        elif selection == '2':
            try:
                display_menu()
                choice = int(input("Enter the number corresponding to the event: "))
                df = event_name(choice)
                if df is not None:
                    try:
                        df['ds'] = pd.to_datetime(df['ds'])
                        start_date_str = input("Enter the first day for ticket sales (YYYY-MM-DD): ")
                        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

                        stop_date_str = input("Enter the last date of the event (YYYY-MM-DD): ")
                        stop_date = datetime.strptime(stop_date_str, '%Y-%m-%d')
                        period(df, stop_date)
                        prediction(df, start_date, stop_date)
                    except ValueError as e:
                        print(f"Date error: {e}")
                        continue
            except ValueError as e:
                print(f"Input error: {e}")
                continue
        elif selection == '3':
            forecast, m = prediction(df, start_date, stop_date)
            plot2(forecast, m)
            continue
        elif selection == '0':
            print("***************************************************************")
            print("Thank you for using Group 3 Prediction App. Have a great day!")
            print("***************************************************************")
            break
        else:
            print("Invalid choice. Please select a valid option.")
            
if __name__ == "__main__":
    main()
