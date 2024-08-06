import os
import csv
import pandas as pd
import datetime

def filterData(input_path, output_path, input_format = "json", overwrite=False, keywords = None, keywords_CS = None,kw_field = None, start_time = None, end_time=None,time_key = None, attrs=None, parse_quotes = True):
    """
    Function to filter a data file by:
    - keywords
    - time range (either start_time or end_time or both)

    Parameters
    ----------
    input_path : str
        Path to the input file
    output_path : str
        Path to the output file
    input_format : json, csv
        Format of the input file
    output_format : json, csv
        Format of the output file
    overwrite : bool
        Flag set to overwrite existing file (if created)
    keywords : list
        List of keywords to filter by
    kw_field : str
        Field to filter by, if not specified, filter by title
    time_key : str
        The time field (str) labelled by the author of the dataset
    attrs : list
        List of attributes to extract from the dataset
    parse_quotes : bool
        Flag to parse quotes in the csv file, as some quoting errors in some of the dataset cause issue with pandas 
    
    """
    pd.options.mode.chained_assignment = None  # default='warn' - supress panda warnings
    file_size = round(os.path.getsize(input_path) / (1024**3),3)
    start_time = pd.to_datetime(start_time).date()
    end_time = pd.to_datetime(end_time).date()
    mode = 'a+' 

    print(f'{file_size} GB')
    # print("Reading:", input_path, "...")

    if not os.path.exists(f"{output_path}"):
        # Overwrite flag is set, creating new file...
        mode = 'w'

    if input_format == "csv":
        quoting = csv.QUOTE_MINIMAL
        if not parse_quotes:
            quoting=csv.QUOTE_NONE
        chunk_iterator = pd.read_csv(input_path, encoding='utf-8', chunksize=200000, on_bad_lines='skip', quoting=quoting)
    else:
        chunk_iterator = pd.read_json(input_path, lines=True, chunksize=200000)

    with open(output_path, mode= mode, newline='', encoding='utf-8') as output_file:
        write_header = True
        if mode == "a+":
            output_file.seek(0)
            first_line = output_file.readline()
            if first_line:
                write_header = False
            output_file.seek(0, 2)

        for chunk in chunk_iterator:
            df = chunk.dropna(subset=[time_key])    # removes any line where created_utc is NaN (no timestamp)

            # the attributes desired from the dataframe i.e., ['title', 'self_text', 'created_utc']
            if attrs:
                df = df[attrs]

            # keywords to filter by either ['title'] or ['self_text'], if not specified, filter by title only
            if keywords:
                if kw_field:
                    df[kw_field] = df[kw_field].str.replace(r'\s+', ' ', regex=True) # remove extra spaces

                    if df[kw_field].isnull().any():
                        # handle missing values for the key being filtered by
                        df[kw_field] = df[kw_field].fillna('')

                    kw_filter = df[kw_field].str.contains('|'.join(keywords), case=False) # creates a list of bools
                    df = df[kw_filter]
                else:
                    kw_filter = df['title'].str.contains('|'.join(keywords), case=False)
                    df = df[kw_filter]

            if start_time or end_time:
                if df[time_key].dtype in ['int64', 'float64']:
                    #timestamp is unix i.e., 1581021943
                    df[time_key] = pd.to_datetime(df[time_key], unit='s')
                # elif df[time_key].dtype == 'object':
                #     for format_string in custom_formats:
                #         try:
                #             df[time_key] = pd.to_datetime(df[time_key], format=format_string)
                #             break
                #         except ValueError:
                #             pass
                elif df[time_key].str.contains('ET').any():
                    # unrecognised ET timezone
                    df[time_key] = df[time_key].str.replace('ET', '', regex=True)
                    df[time_key] = pd.to_datetime(df[time_key]).dt.tz_localize('US/Eastern')
                else:
                    #timestamp is ISO, or different, let panda parse
                    df[time_key] = pd.to_datetime(df[time_key])

                if df[time_key].dt.tz:
                    # some time objects timestamp are localized, which the start and end time cannot be compared with
                    df[time_key] = df[time_key].dt.tz_localize(None)
                

                df[time_key] = df[time_key].dt.date

                if start_time and end_time:
                    df = df[(df[time_key] >= start_time) & (df[time_key] <= end_time)]

                elif start_time:
                    df = df[df[time_key] >= start_time]

                elif end_time:
                    df = df[df[time_key] <= end_time]

            df.rename({kw_field: 'text',time_key: 'date_utc'}, axis=1, inplace=True)
            df.to_csv(output_file, mode=mode, header=write_header, index=False, encoding='utf-8')
            mode = 'a'

if __name__ == "__main__":
    """
    Example usage of the filterData function
    """
    input_path = "extracted_wsb_dumps"
    output_path = "data/wsb_submissions_csv/"
    keys = ["Apple", "Tesla"]
    for file in os.listdir(input_path):
        print("processing " + f"{input_path}\{file}")
        keys_string = '_'.join(keys)
        filterData(f"{input_path}\{file}", output_path, f"{file}_{keys_string}", overwrite=True, keywords=keys, kw_field='title', start_time=datetime(2021, 1, 1), end_time=datetime(2022, 12, 31), attrs=['title', 'created_utc', 'url'])

    print("All files processed")


