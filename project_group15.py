# Importing libraries
import pandas as pd
import numpy as np


# Reading data files
employees_df = pd.read_csv('employees.csv')
attendance_df = pd.read_csv('attendance.csv')
holidays_df = pd.read_csv('holidays.csv')
leaves_df = pd.read_csv('leaves.csv')
salary_df = pd.read_csv('salary.csv')



#-----------------------------------------------------------------
#----------------Preprocessing 'attendance.csv'-------------------
#-----------------------------------------------------------------

# Dropping unnecessary columns
columns_to_drop = ['id', 'date', 'out_date']
attendance_df.drop(columns=columns_to_drop, inplace=True)

# Dropping records of employees who are not mentioned in the employees.csv table
unique_employee_attendance = attendance_df['Employee_No'].unique()
unique_employee_employee = employees_df['Employee_No'].unique()

employee_not_in_employee_df = set(unique_employee_attendance) - set(unique_employee_employee)
attendance_df_filtered = attendance_df[~attendance_df['Employee_No'].isin(employee_not_in_employee_df)]


# Convert 'Hourly_Time' column to numeric data type
attendance_df_filtered['Hourly_Time'] = pd.to_numeric(attendance_df_filtered['Hourly_Time'], errors='coerce')

# Outlier detection
q1 = attendance_df_filtered['Hourly_Time'].quantile(0.25)
q3 = attendance_df_filtered['Hourly_Time'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 3 * iqr
upper_bound = q3 + 3 * iqr
outliers = attendance_df_filtered[(attendance_df_filtered['Hourly_Time'] < lower_bound) | (attendance_df_filtered['Hourly_Time'] > upper_bound)]
num_outliers = len(outliers)

# Filtering out negative values and extreme outliers
attendance_df_filtered = attendance_df_filtered[(attendance_df_filtered['Hourly_Time'] >= 0) & (attendance_df_filtered['Hourly_Time'] <= 24)]


# Sorting data in ascending order of employee numbers
attendance_df_filtered = attendance_df_filtered.sort_values(by='Employee_No', ascending=True)

# Obtaining the projects a given employee has worked on
grouped = attendance_df_filtered.groupby('Employee_No')['project_code'].agg(set)
project_codes = pd.DataFrame(grouped).reset_index()
project_codes.columns = ['Employee_No', 'Project_Codes']

# Obtaining Hourly_time= 0 as absent and getting the count per each employee
filtered_df = attendance_df_filtered[attendance_df_filtered['Hourly_Time'] == 0]
absent_counts = filtered_df.groupby('Employee_No')['Hourly_Time'].count().reset_index()
absent_counts.columns = ['Employee_No', 'absent count']

attendance_df_filtered = attendance_df_filtered[attendance_df_filtered['Hourly_Time'] != 0]

# Obtaining the late minutes by taking the difference of in_time and Shift_Start
attendance_df_filtered['in_time'] = pd.to_datetime(attendance_df_filtered['in_time'])
attendance_df_filtered['Shift_Start'] = pd.to_datetime(attendance_df_filtered['Shift_Start'])
attendance_df_filtered['late_minutes'] = (attendance_df_filtered['in_time'] - attendance_df_filtered['Shift_Start']).dt.total_seconds() / 60

# Obtaining the leave early minutes by taking the difference of out_time and Shift_End
attendance_df_filtered['out_time'] = pd.to_datetime(attendance_df_filtered['out_time'])
attendance_df_filtered['Shift_End'] = pd.to_datetime(attendance_df_filtered['Shift_End'])
attendance_df_filtered['leave_early_minutes'] = (attendance_df_filtered['Shift_End'] - attendance_df_filtered['out_time']).dt.total_seconds() / 60

# Dropping unnecessary columns
attendance_df_filtered.drop(['in_time', 'out_time', 'Shift_Start', 'Shift_End'], axis=1, inplace=True)

attendance_df_filtered['late_hours'] = round(attendance_df_filtered['late_minutes'] / 60, 2)
attendance_df_filtered['leave_early_hours'] = round(attendance_df_filtered['leave_early_minutes'] / 60, 2)
attendance_df_filtered.drop(columns=['late_minutes', 'leave_early_minutes'], inplace=True)

# Combining all the records of a single employee to get average values
attendance_df_filtered = attendance_df_filtered.drop(columns=['project_code'])

attendance_final = attendance_df_filtered.groupby('Employee_No').agg({
    'Hourly_Time': 'mean',
    'late_hours': 'mean',
    'leave_early_hours': 'mean'
}).reset_index()

attendance_final = attendance_final.round({'Hourly_Time': 2, 'late_hours': 2, 'leave_early_hours': 2})
attendance_final.columns = ['Employee_No', 'Average_work_Time', 'Average_late_hours', 'Average_leave_early_hours']

# Merging the dataframes

# Adding the project codes to the main dataframe
attendance_final = pd.merge(attendance_final, project_codes, on='Employee_No', how='left')

# Adding the absent count to the main dataframe
attendance_final = pd.merge(attendance_final, absent_counts, on='Employee_No', how='left')
attendance_final['absent count'] = attendance_final['absent count'].fillna(0).astype(int)  # Final preprocessed attendance dataframe

print("Attendance data preprocessing completed successfully")



#-----------------------------------------------------------------
#----------------Preprocessing 'salary.csv'---------------------
#-----------------------------------------------------------------

# Selecting numeric features
salary_df = salary_df.select_dtypes(include=['number'])

# These features were selected by training a random forest regressor and selecting the most important features
selected_df = salary_df.loc[:, ["Employee_No","Total Earnings_2", "Net Salary", "Total Deduction"]]

# Filter out the cases where Net Salary not equals Total Earnings_2 - Total Deduction
filtered_df = selected_df[selected_df['Net Salary'] != selected_df['Total Earnings_2'] - selected_df['Total Deduction']]

# Calculate the difference between Net Salary and Total Earnings_2
filtered_df['Difference'] = abs(filtered_df['Net Salary'] - (filtered_df['Total Earnings_2'] - filtered_df['Total Deduction']))
filtered_df['Difference'] = filtered_df['Difference'].astype(int)

# Sort the DataFrame by the difference in descending order
filtered_df = filtered_df.sort_values(by='Difference', ascending=False)

# temporarly removing all the cases where either Net Salary or Total Earnings become zero
filtered_df_new = filtered_df[(filtered_df['Net Salary'] != 0) & (filtered_df['Total Earnings_2'] != 0)]

# Drop the cases where 'Difference' is zero
filtered_df_new = filtered_df_new[filtered_df_new['Difference'] != 0]
filtered_df_new = filtered_df_new[filtered_df_new['Difference'] >= 1000]


# only filtering out the necessary values for the final insight ( net salary)
salary_df = salary_df.loc[:, ["Employee_No","Total Earnings_2", "Net Salary", "Total Deduction"]]

# checking for differences
employee_no_unique = set(employees_df['Employee_No'].unique())
salary_employee_no_unique = set(salary_df['Employee_No'].unique())

employee_no_only_in_employee_df = employee_no_unique - salary_employee_no_unique
employee_no_only_in_salary_df = salary_employee_no_unique - employee_no_unique

employee_no_only_in_salary_df  = list(employee_no_only_in_salary_df )
# Dropping rows based on Employee_No values
salary_df = salary_df[~salary_df['Employee_No'].isin(employee_no_only_in_salary_df)]

unique_employee_nos = salary_df['Employee_No'].unique()
unique_employee_nos.sort()

# Replace the values of "Net Salary" with "Total Earnings_2" where "Net Salary" is 0
salary_df.loc[salary_df['Net Salary'] == 0, 'Net Salary'] = salary_df.loc[salary_df['Net Salary'] == 0, 'Total Earnings_2']

salary_df = salary_df[salary_df['Net Salary'] != 0]

salary_df_clean = salary_df[~salary_df.index.isin(filtered_df_new.index)]

# Resetting the index if needed
salary_df_clean.reset_index(inplace=True)

# 2. Take the average value of the Net salary column for entries with the same Employee_No value
average_salary = salary_df_clean.groupby('Employee_No')['Net Salary'].mean().reset_index()

# Convert the average salary to integer
average_salary['Net Salary'] = average_salary['Net Salary'].astype(int)

# Rename the 'Net Salary' column to 'Average Salary'
average_salary.rename(columns={'Net Salary': 'Average Salary'}, inplace=True)   # Final preprocessed Salary dataframe

print("Salary data preprocessing completed successfully")


#-----------------------------------------------------------------
#----------------Preprocessing 'leaves.csv'---------------------
#-----------------------------------------------------------------

# Filter leaves_df to keep only rows where Employee_No is in valid_employee_nos
valid_employee_nos = set(employees_df['Employee_No'])
leaves_df = leaves_df[leaves_df['Employee_No'].isin(valid_employee_nos)]

# Replace specified values with 1
values_to_replace = ['Personal', 'Having a wedding to attend', 'For Covid Vaccination.', 'For a personal matter',
                     'Personal Matter', 'Went to meet the Doctor', 'to be taken 2nd dos vaslen', 'Vaccination-2nd Dose',
                     'Personnel', 'Went to the dental Hospital', 'Sick', 'Went to get the Cov19 Vaccine', 'COVID 19',
                     'PERSONAL', 'For a personal Matter', 'Personal Reason', 'Personel', 'Personal (house shifting)',
                     'For personal matter', 'Short leave on 14.11 & 20.11', 'Short leave on 20.12 & 24.12',
                     'Short leaves on 26.11 & 15.12', 'personal', 'Attending the Arbitration cour', 'Attending Class',
                     'Personnal', 'Medical Leave', 'Fever and diarrhoea', 'Medical for  VISA', 'COurt Case',
                     'Sick Leave', 'attend classes', 'New Year leave', 'Company requirement',
                     'Transport issue due to Fuel sh', 'personel matter', 'Curfew & Transport Issue', 'Transport Issue',
                     'Curfew Leave', 'Sickness- Virus Fever', 'SUFFERING FROM FEVER', 'DUE TO AN ALMS GIVING',
                     'For personal Matter', 'Funeral', 'Go to meet doctor', 'For Personal Matter', 'no transport available',
                     'personals', 'For Sickness', 'Halfday-( 10:00 - 14:00)', 'for my wedding', 'For my wedding',
                     'For Personal matter', 'For a personal matter.', 'Suffering from back pain', 'went to chilaw',
                     "Daughter's Convocation", 'Wedding', "Mother's Arms giving", 'Headache & Faintness', 'Personal Mater',
                     'Not Well', 'Not well', 'Duty Leave', 'Birthday - Muruthen Poojawa.', 'Arms Giving',
                     "3 months's Bana", "3 Month's Arms Giving", "Dad's Arms Giving", 'Covid vaccine 2nd dose',
                     'For Covid 19 Vaccination', "Due to father's ankle surgery", 'Vaccine 2nd Doze', 'Final Viva Exam',
                     'To attend IESL Examination P1', 'To attend IESL Examination P2', 'TO GET THE BOOSTER VACCINE',
                     'pursonal mator', 'For Taking COVID Vaccine (3rd)', 'Personal (Wedding)', 'For a Personal Matter',
                     'due to a Sickness', 'Personal work at Home town', 'To get the vaccine', 'Attending lectures',
                     'Fewer', 'Fever', 'Emergency', 'reson', 'Perssonal', 'personnal', 'perssonal', 'cough and fever',
                     'For new year Festival', 'For NewYear', 'New Year', 'Presonal', 'a Personal matter', 'pERSONAL',
                     'Site Shutdown for new year', 'EID FESTIVAL', 'Office Closed.', 'Personal work (1pm onwards)', '.',
                     'For a personal matter0', 'Curfew', 'For sickness', 'Vehicle breakdown', 'sick',
                     'Due to fuel shortage', 'Personal work', 'Personal matter', 'Due to the lack of Transport',
                     'Personal.', 'Personal Work', 'Visa process', 'Personal  Matter', 'NOT WELL', 'Suffering from fever',
                     'not well', 'Sick & Not Well', 'SICK', 'personal mater', 'Preparing for IELTS Exam', 'private',
                     'Went to Colombo', 'For channel doctor', 'To attend Police Station', 'Attend for urgent matter',
                     'Went to Baththramulla for urgent matter', 'personal work', 'Due to Illness', 'For exam', 'Peronal']

leaves_df['Reason_provided'] = leaves_df['Remarks'].replace(values_to_replace, '1')

# Replace non-'0' and non-'1' values with 0
leaves_df['Reason_provided'] = leaves_df['Reason_provided'].replace({'\\N': '0', 'nan': '0'})

leaves_df.fillna("0", inplace=True)

leaves_df.drop(columns=['Remarks'], inplace=True)

# Calculate the difference in days between the applied date and the actual leave date
leaves_df['leave_date'] = pd.to_datetime(leaves_df['leave_date'])
leaves_df['Applied Date'] = pd.to_datetime(leaves_df['Applied Date'].str.split().str[0])
# Calculate the difference in days
leaves_df['days_between'] = (leaves_df['Applied Date'] - leaves_df['leave_date']).dt.days

leaves_df = leaves_df.drop(['leave_date', 'Applied Date'], axis=1)

# Drop the rows where the days_between value is above 600
leaves_df = leaves_df[leaves_df['days_between'] <= 600]
leaves_df = leaves_df[leaves_df['days_between'] > 0]

leaves_df = leaves_df.sort_values(by='Employee_No')

aggregated_df = leaves_df.groupby('Employee_No').agg(
    Half_Day_Count=pd.NamedAgg(column='Type', aggfunc=lambda x: (x == 'Half Day').sum()),
    Full_Day_Count=pd.NamedAgg(column='Type', aggfunc=lambda x: (x == 'Full Day').sum()),
    Anual_Count=pd.NamedAgg(column='apply_type', aggfunc=lambda x: (x == 'Anual').sum()),
    Casual_Count=pd.NamedAgg(column='apply_type', aggfunc=lambda x: (x == 'Casual').sum())
)

# Displaying only one row per unique employee number
aggregated_df.reset_index(inplace=True)  # Resetting index to display 'Employee_No' as a column
aggregated_df = aggregated_df.sort_values(by='Employee_No')                     # Final preprocessed Leaves dataframe

print('Leaves data preprocessing completed successfully')


#-----------------------------------------------------------------
#----------------Preprocessing 'employees.csv'---------------------
#-----------------------------------------------------------------

# Dropping unnecessary columns
employees_df.drop(columns = ['Employee_Code', 'Name', 'Religion_ID', 'Designation_ID', 'Reporting_emp_1', 'Reporting_emp_2'], inplace = True)

# Replacing '0000' in Year_of_Birth with NA values
employees_df.loc[employees_df['Year_of_Birth'] == "'0000'", 'Year_of_Birth'] = pd.NA

# drop the duplicate records if there are any
employees_df.drop_duplicates()

for index, row in employees_df.iterrows():
    gender = row['Gender']
    title = row['Title']

    # Change title to 'Mr' for male employees with titles 'Ms' or 'Miss'
    if ((gender == 'Male') & (title != 'Mr')):
      employees_df.at[index, 'Title'] = 'Mr'

    # Change title to 'Miss' for single female employees with titles 'Ms' or 'Mr'
    if ((gender == 'Female') & (title == 'Single') & (title != 'Miss')):
      employees_df.at[index, 'Title'] = 'Miss'

    # Change title to 'Miss' for married female employees with titles 'Miss' or 'Mr'
    if ((gender == 'Female') & (title == 'Married') & (title != 'Ms')):
      employees_df.at[index, 'Title'] = 'Ms'

# Refining the Date_resigned, Inactive_Date and Status columns
for index, row in employees_df.iterrows():
    status = row['Status']
    date_resigned = row['Date_Resigned']
    inactive_date = row['Inactive_Date']

    if (status == 'Active'):
        employees_df.at[index, 'Date_Resigned'] = '\\N'
        employees_df.at[index, 'Inactive_Date'] = '\\N'

    elif ((date_resigned == '0000-00-00') and (inactive_date != '0000-00-00')):
        employees_df.at[index, 'Date_Resigned'] = inactive_date

    elif ((inactive_date == '0000-00-00') and (date_resigned != '0000-00-00')):
        employees_df.at[index, 'Inactive_Date'] = date_resigned

    elif ((inactive_date == '0000-00-00') and (date_resigned == '0000-00-00')):
        employees_df.at[index, 'Date_Resigned'] = '\\N'
        employees_df.at[index, 'Inactive_Date'] = '\\N'
        employees_df.at[index, 'Status'] = 'Active'


### Preparing for KNN imputation

# Extracting sub columns of Marvellous dataframe for performing KNN imputation
df_sub = employees_df[['Gender', 'Marital_Status', 'Date_Joined', 'Date_Resigned', 'Year_of_Birth']]

# Extracting Year_Joined from the Date_Joined
df_sub['Year_Joined'] = df_sub['Date_Joined'].apply(lambda x: x[-4:])

# Extracting Year_Resigned from the Date_Resigned
def extract_year(x):
  if  x != "\\N":
    return x[-4:]
  return 0
df_sub['Year_Resigned'] = df_sub['Date_Resigned'].apply(extract_year)

# dropping date columns as we got enough information out of these columns for our imputation
df_sub.drop(columns = ['Date_Joined', 'Date_Resigned'], inplace = True)

cols = ['Gender', 'Marital_Status', 'Year_Joined', 'Year_Resigned', 'Year_of_Birth']
df_sub = df_sub[cols]

# convering string data type in to numerical data
marital_sts = {'Single':0, 'Married':1}
df_sub['Marital_Status'] = df_sub['Marital_Status'].map(marital_sts)
df_sub.rename(columns={'Marital_Status':'Marital_Status'}, inplace=True)

gender = {'Female':0, 'Male':1}
df_sub['Gender'] = df_sub['Gender'].map(gender)
df_sub.rename(columns={'Gender':'Gender'}, inplace=True)

df_sub.replace(pd.NA, np.nan, inplace=True)

###### KNN Imputation
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors = 10)
imputed_data = knn_imputer.fit_transform(df_sub)
new_imputed = pd.DataFrame(imputed_data)

i=0
for col in new_imputed.columns:
  new_imputed.rename(columns = {col:cols[i]}, inplace=True)
  i+=1

for index, row in new_imputed.iterrows():
  new_imputed.loc[index, 'Year_of_Birth'] = round(row['Year_of_Birth'])

df_sub = new_imputed

# Changing the Marital_Status to its original string domain
for index, row in df_sub.iterrows():
  married = row['Marital_Status']

  if married:
    df_sub.loc[index, 'Marital_Status'] = "Married"
  else:
    df_sub.loc[index, 'Marital_Status'] = "Single"

# Extracting Necessary, Imputed columns from df_sub data frame into Marvellous DataFrame.
employees_df['Year_of_Birth'] = df_sub['Year_of_Birth']
employees_df['Marital_Status'] = df_sub['Marital_Status']    # Final preprocessed employee dataframe


print("Employees data preprocessing completed successfully")


#---------------------------------------------------------------------------------------------------
#----------------Merging the useful columns obtained by processing other tables---------------------
#---------------------------------------------------------------------------------------------------

# Merging the dataframes
employees_df = pd.merge(employees_df, attendance_final, on='Employee_No', how='left')
employees_df.fillna('NA', inplace=True)

employees_df = pd.merge(employees_df, average_salary, on='Employee_No', how='left')
employees_df.fillna('NA', inplace=True)

employees_df.to_csv('employee_preprocess_project_group15.csv', index=False)





