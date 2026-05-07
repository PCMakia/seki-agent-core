import win32com.client
import datetime

calendar = win32com.client.Dispatch("Outlook.Application")

# Define the event
subject = "Test Event"
body = "Characteristic of event"
start_date = datetime.datetime.now()

duration = datetime.timedelta(hours=1)

# Create the event in calendar
appointment = calendar.CreateItem(1) # 1 represents the AppointmentItem

# Set event details
appointment.Subject = subject
appointment.Body = body
appointment.Start = start_date
appointment.Duration = duration

appointment.ReminderSet = True
appointment.ReminderMinutesBeforeStart = 15

# Save the event to the calendar
appointment.Save() 

print("Event created successfully")