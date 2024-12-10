import pandas as pd

def clean_df(df):
    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    # Drop the first two rows (weird headers)
    df = df.drop(index=[0, 1])
    # Drop empty rows
    df = df.dropna(axis=0, how='all')

    # Drop columns not used in analysis
    # Includes explicitly dropping empty columns
    columns_to_drop = ['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress',
                'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId',
                'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
                'ExternalReference', 'LocationLatitude', 'LocationLongitude',
                'DistributionChannel', 'UserLanguage', 'Q_RecaptchaScore',
                'Q26', 'Q6_4_TEXT', 'Q25 - Topics', 'Q25 - Parent Topics',
                'Q18 - Topics', 'Q18 - Parent Topics', 'Q25 - Topic Hierarchy Level 1',
                'Q18 - Topic Hierarchy Level 1'
                ]

    df = df.drop(columns=columns_to_drop)

    # Give columns relevant names based on the question
    column_names_clean = [
        # TEXT = Written Response from user
        # Q1 is consent question
        'Age', # Q2
        'Gender', # Q3
        'Education', 'Education_TEXT', #Q4
        'ScreenTime', #Q5
        'Confidence', #Q6
        'Terms&Conditions', #Q7
        'ReviewPermissions', #Q8
        'Cookies', #Q9
        'FamiliarDP',#Q10
        'DefineDP_TEXT', # Q10.1. (Optional)
        'PressureCreateAccount', # Q11
        'PressureCreateAccountFeel_TEXT', # Q11.1. (Optional)
        'DeleteAccount', # Q12
        'DeleteAccountExperience_TEXT', # Q12.1. (Optional)
        'DeceptivePlatformsAndServices_TEXT', # Q13
        'DataRemovalServiceUse', # Q14
        'DataRemovalServiceMotivation_TEXT', # Q14.1. (Optional)
        'WhichDataRemovalService',  'WhichDataRemovalService_TEXT', # Q14.2. (Optional)
        'CCPA', # Q15
        'UsedLaws' # Q15.1
    ]

    # Create a dictionary mapping old column names to new column names
    rename_dict = dict(zip(df.columns, column_names_clean))

    #  # Rename the columns
    df = df.rename(columns=rename_dict)

    # Fill NaN values with 0 in all columns
    df.fillna(value=0, inplace=True)

    return df

def main():
    csv_file_path = 'Survey_December 8, 2024_20.21.csv'
    df = pd.read_csv(csv_file_path)

    df = clean_df(df)
    print(df.columns)
    print(df.info)

    # Save the cleaned DataFrame as a pickle file
    # To avoid the need to create the dataframe every time the analysis is run
    df.to_pickle('survey_12-8.pkl')
    exit()

if __name__ == "__main__":
    main()