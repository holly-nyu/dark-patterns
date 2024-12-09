import pandas as pd


def clean_df(df):
    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    # Drop the first two rows
    df = df.drop(index=[0, 1])

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
        'Q4_Age_C',
        'Q6_Gender_C',
        'Q7_Education_C', 'Q7_Education_TEXT',
        'Q8_ScreenTime_C',
        'Q20_Confidence_R',
        'Q21-1_Terms&Conditions_R', 'Q21-2_ReviewPermissions_R', 'Q21-3_Cookies_R',
        'Q22_FamiliarDP_YN',
        'Q24_DefineDP_O_TEXT',
        'Q23_PressureCreateAccount_YN',
        'Q23_PressureCreateAccountFeel_O_TEXT',
        'Q16_DeleteAccount_YN',
        'Q16_DeleteAccountExperience_O_TEXT',
        'Q18_DeceptivePlatformsAndServices_TEXT',
        'Q25_DataRemovalServiceUse_YN',
        'Q23_DataRemovalServiceMotivation_O_TEXT',
        'Q19_WhichDataRemovalService_O_C',
        'Q19_WhichDataRemovalService_O_TEXT',
        'Q20_CCPA_YN',
        'Q21_UsedLaws_O'
    ]

   # Create a dictionary mapping old column names to new column names
    rename_dict = dict(zip(df.columns, column_names_clean))

    #  # Rename the columns
    df = df.rename(columns=rename_dict)

    # Custom sorting function
    def sort_key(col):
        parts = col.split('_')[0].split('-')
        return (int(parts[0][1:]), int(parts[1]) if len(parts) > 1 else 0)

    # Sort columns
    sorted_columns = sorted(df.columns, key=sort_key)

    # Reorder DataFrame columns
    df = df[sorted_columns]

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