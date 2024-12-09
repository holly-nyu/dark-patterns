import pandas as pd


def clean_df(df):
    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    # Drop the first two rows (weird headers)
    df = df.drop(index=[0, 1])
    # Drop empty rows
    df = df.dropna(axis=0, how='all')


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

    column_names_clean = [
        # C = Choice
        # TEXT = Written Response from user
        # R = Rank
        'Age',
        'Gender',
        'Education', 'Education_TEXT',
        'ScreenTime',
        'Confidence',
        'Terms&Conditions', 'ReviewPermissions', 'Cookies',
        'FamiliarDP',
        'DefineDP_TEXT',
        'PressureCreateAccount',
        'PressureCreateAccountFeel_TEXT',
        'DeleteAccount',
        'DeleteAccountExperience_TEXT',
        'DeceptivePlatformsAndServices_TEXT',
        'DataRemovalServiceUse',
        'DataRemovalServiceMotivation_TEXT',
        'WhichDataRemovalService',
        'WhichDataRemovalService_TEXT',
        'CCPA',
        'UsedLaws'
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
    df.to_pickle('survey_12-8.pkl')
    exit()

if __name__ == "__main__":
    main()