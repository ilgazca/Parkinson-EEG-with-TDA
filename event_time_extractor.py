import pandas as pd

# List of the .tsv files you want to process
file_list = [
    #"/home/ilgazc/Data/Parkinson_medOn_medOff/AbzsOg/sub-AbzsOg_ses-PeriOp_task-HoldL_acq-MedOff_run-1_split-01_events.tsv",
    "/home/ilgazc/Data/Parkinson_medOn_medOff/FYbcap/sub-FYbcap_ses-PeriOp_task-HoldL_acq-MedOn_run-1_events.tsv"
]

# A dictionary to store all the dataframes
all_events_data = {}

print("Starting to process event files...\n")

for file_name in file_list:
    print(f"--- Processing File: {file_name} ---")

    try:
        # Use pandas.read_csv, specifying the separator as a tab ('\t')
        events_df = pd.read_csv(file_name, sep='\t')

        # Convert onset and duration columns to numeric data types
        # This is necessary for doing math (like addition)
        events_df['onset'] = pd.to_numeric(events_df['onset'])
        events_df['duration'] = pd.to_numeric(events_df['duration'])

        # Calculate the 'end' time by adding duration to onset
        events_df['end'] = events_df['onset'] + events_df['duration']

        # Re-order the columns to match the table I showed you
        events_df = events_df[['trial_type', 'onset', 'duration', 'end']]

        # Store the dataframe in our dictionary
        all_events_data[file_name] = events_df

        # Print the final table to your console
        # .to_string(index=False) makes it look clean
        print(events_df.to_string(index=False))
        print("\n")

    except FileNotFoundError:
        print(f"ERROR: File not found: '{file_name}'")
        print("Please make sure the file is in the same directory as the script.\n")
    except Exception as e:
        print(f"An error occurred processing {file_section}: {e}\n")

print("...Processing complete.")

# You can now access any of the dataframes by its filename, for example:
# if 'sub-AbzsOg_ses-PeriOp_task-HoldL_acq-MedOn_run-2_events.tsv' in all_events_data:
#     print("\nAccessing data from run-2:")
#     print(all_events_data['sub-AbzsOg_ses-PeriOp_task-HoldL_acq-MedOn_run-2_events.tsv'])