import tkinter as tk
from tkinter import messagebox, scrolledtext
from datetime import datetime
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import os

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Group 3 Prediction App")
        self.center_window(840, 750)  # Center the window with the specified width and height
        self.root.resizable(True, True)  # Make the window resizable
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.selected_df = None
        self.setup_ui()

    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        tk.Label(self.root, text="Welcome to the Prediction App, how can we assist you today?", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        tk.Button(self.root, text="1. Display list of Events Present", command=self.display_events, width=30).grid(row=1, column=0, columnspan=2, pady=10)
        self.predict_button = tk.Button(self.root, text="2. Do a Prediction", command=self.show_prediction_fields, state="disabled", width=30)
        self.predict_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_date_label = tk.Label(self.root, text="Enter the first day for ticket sales (YYYY-MM-DD):", anchor='e')
        self.start_date_entry = tk.Entry(self.root, width=30)
        self.stop_date_label = tk.Label(self.root, text="Enter the last date of the event (YYYY-MM-DD):", anchor='e')
        self.stop_date_entry = tk.Entry(self.root, width=30)

        self.do_prediction_button = tk.Button(self.root, text="Predict", command=self.perform_prediction, width=30)
        
        self.output_display = scrolledtext.ScrolledText(self.root, width=100, height=20, state="disabled")
        self.output_display.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        tk.Button(self.root, text="0. Quit", command=self.quit_app, width=30).grid(row=7, column=0, columnspan=2, pady=10)

    def load_and_concatenate(self, selected_files):
        dataframes = []
        for file in selected_files:
            try:
                dataframes.append(pd.read_csv(file))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {file}: {e}")
                return None
        df = pd.concat(dataframes, axis=0)
        df = df.rename(columns={'Created Date': 'ds', 'Attendance Count': 'y'})
        return df.reset_index(drop=True)

    def event_name(self, name):
        events = {
            1: [os.path.join(self.current_directory, 'D19_encoded.csv'), os.path.join(self.current_directory, 'D21_encoded.csv'), self.latest_event()],
            2: [os.path.join(self.current_directory, 'GP21_encoded.csv'), os.path.join(self.current_directory, 'NP21_encoded.csv'), self.latest_event()],
            3: [os.path.join(self.current_directory, 'SRM22_encoded.csv'), self.latest_event()],
            4: self.all_data_sets()
        }
        return self.load_and_concatenate(events.get(name, []))

    def latest_event(self):
        return os.path.join(self.current_directory, 'SRM23_test_encoded.csv')

    def all_data_sets(self):
        return [
            os.path.join(self.current_directory, 'D19_encoded.csv'),
            os.path.join(self.current_directory, 'D21_encoded.csv'),
            os.path.join(self.current_directory, 'GP21_encoded.csv'),
            os.path.join(self.current_directory, 'MSE21_encoded.csv'),
            os.path.join(self.current_directory, 'NP21_encoded.csv'),
            os.path.join(self.current_directory, 'SRM22_encoded.csv'),
            self.latest_event()
        ]

    def display_events(self):
        def on_select_event():
            choice = int(event_var.get())
            event_selection_window.destroy()
            df = self.event_name(choice)
            if df is not None:
                output_text = f"{Event_list[choice]} dataset uploaded\n\n"
                output_text += "First few rows:\n"
                output_text += df.head().to_string(index=False)
                output_text += "\n\nLast few rows:\n"
                output_text += df.tail().to_string(index=False)
                output_text += "\n****************"
                
                self.output_display.config(state="normal")
                self.output_display.delete(1.0, tk.END)
                self.output_display.insert(tk.END, output_text)
                self.output_display.config(state="disabled")

                self.predict_button.config(state="normal")
                self.selected_df = df
            else:
                messagebox.showerror("Error", "Invalid event selection.")
        
        event_selection_window = tk.Toplevel(self.root)
        event_selection_window.title("Choose an Event")
        event_selection_window.geometry("400x250")
        tk.Label(event_selection_window, text="Please choose an event:", font=("Helvetica", 14)).pack(pady=10)
        event_var = tk.StringVar(value="1")
        for key, value in Event_list.items():
            tk.Radiobutton(event_selection_window, text=value, variable=event_var, value=key).pack(anchor="w")
        tk.Button(event_selection_window, text="Select", command=on_select_event, width=20).pack(pady=10)

    def show_prediction_fields(self):
        self.start_date_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.start_date_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        self.stop_date_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.stop_date_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        self.do_prediction_button.grid(row=5, column=0, columnspan=2, pady=10)

    def perform_prediction(self):
        try:
            df = self.selected_df
            df['ds'] = pd.to_datetime(df['ds'])
            start_date = datetime.strptime(self.start_date_entry.get(), '%Y-%m-%d')
            stop_date = datetime.strptime(self.stop_date_entry.get(), '%Y-%m-%d')
            forecast, m = self.make_prediction(df, start_date, stop_date)
            self.plot_forecast(forecast, m)
        except ValueError as e:
            messagebox.showerror("Error", f"Input error: {e}")

    def make_prediction(self, df, start_date, stop_date):
        m = Prophet()
        m.fit(df)
        prediction_period = self.calculate_period(df, stop_date)
        future = m.make_future_dataframe(periods=prediction_period)
        future = future[future['ds'] > df['ds'].max()]
        forecast = m.predict(future)
        forecast = forecast[forecast['yhat'] > 0]
        Predicted_results = forecast['yhat'].sum()
        TrueDate_Ranges = df[df['ds'] >= start_date]
        sold_todate = TrueDate_Ranges['y'].sum()
        sum_y_actual = sold_todate + Predicted_results
        result_str = f'\nThe number of tickets already sold is: {sold_todate}\n''\n' \
                     f' The number of days from now to the event date is : {prediction_period} days \n''\n' \
                     f'From today till the event day, the number of tickets that may be sold given the current trajectory is: {round(Predicted_results)}\n''\n' \
                     f'The Total number of tickets to be sold given the current trajectory is: {round(sum_y_actual)}\n \n'
        messagebox.showinfo("Prediction Result", result_str)
        return forecast, m

    def calculate_period(self, df, stop_date):
        last_date = df['ds'].max()
        return (stop_date - last_date).days

    def plot_forecast(self, forecast, m):
        fig1 = m.plot(forecast)
        ax = fig1.gca()
        ax.set_title('Predicted Ticket Sales')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Tickets Sold')
        plot_path = os.path.join(self.current_directory, 'forecast_plot.png')
        plt.savefig(plot_path)
        try:
            os.startfile(plot_path)  # Use os.startfile for Windows
            messagebox.showinfo("Plot Saved", f"Plot saved successfully to {plot_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open plot: {e}")

    def quit_app(self):
        self.root.destroy()

# List of different types of events
Event_list = {1: "IT Managers", 2: "Property Managers", 3: "Education Managers", 4: "All the Events"}

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
