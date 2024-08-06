from datetime import datetime
import os
import model
from utils import *
from filterData import filterData
from tickers import getClose
from datetime import timedelta, datetime
import time
from evaluate import Evaluate
from xlsxwriter.utility import xl_col_to_name

def main():
    """
    Firstly, we need to extract the related titles that mentions keywords 
    for each ticker. There're multiple datasets available, most of which are 
    raw data (JSON) and need to be process & filter. They're divided into 2 
    types: 
    - News-based: e.g., Bloomberg, Reuters, etc.
    - Media-based: e.g., Reddit, Twitter, etc.

    List of tickers that we're interested in:
        Apple, AAPL
        Tesla, TSLA
        Amazon, AMZN
        Microsoft, MSFT
        Google, GOOGL
        Netflix, NFLX

    We first process the media-based data, which is located in data/raw/media-based.
    """

    m = model.model()
    m.batch_size = 1000 # no. lines to be processed at once, e.g.,a batch size of 1000 means 1000 lines will be read at a time

    worddict = [
        ['AAPL', 'Apple'],
        ['TSLA', 'Tesla'],
        ['AMZN', 'Amazon'],
        ['MSFT', 'Microsoft'],
        ['META','Facebook' ,'FB']]
        # ['Nvidia', 'NVDA'],
        # ['Eil Lilly', 'LLY'],
        # ['Broadcom', 'AVGO'],
        # ['Jp Morgan', 'JPM']]

    media_path = "data/raw/media-based"
    output_media_path = "data/processed/media-based"
    output_mmodel_path = "data/model_results/media-based"

    start = '2016-03-02' # the first date to be included in the data
    end = '2020-12-30' # the last date to be included in the data
    c = False # flag - wipe all existing outputs

    if c and os.path.exists(output_media_path):
        print(f"Cleaning {output_media_path}...")
        clean([output_media_path, output_mmodel_path])

    keysdict = [
        #[time_key, kw_field]
        ['created_utc', 'title', True], # extracted wsb data
        ['timestamp', 'text', False], # financial tweets
    ]

    for i, dir in enumerate(os.listdir(media_path)):
        input_dir = f"{media_path}/{dir}" # path to the extracted data
        output_dir = f"{output_media_path}/{dir}" # path to output the filtered data
        time_key = keysdict[i][0] # the col name for the timestamp in the data
        kw_field = keysdict[i][1] # the field to filter by, e.g., title, self_text, etc.
        attrs = [time_key, kw_field] # the attributes to be extracted from the data
        quote = keysdict[i][2] # flag to parse quotes
        counts = []

        for keys in worddict:
            print()
            print(f"-----------------Extracting for {output_dir}/{keys[0]}-----------------")
            print()

            file_out_path = f"{output_dir}/{keys[0]}({datetime.strptime(start,'%Y-%m-%d').strftime('%d.%m.%Y')}--{datetime.strptime(end,'%Y-%m-%d').strftime('%d.%m.%Y')}).csv"

            if os.path.exists(file_out_path):
                print(f"{file_out_path} already exists, skipping...")
                continue
            
            for file in os.listdir(input_dir):
                file_in_path = f"{input_dir}/{file}"
                format = "json" # the format of the data, e.g., json or csv
                if file.endswith(".csv"):
                    format = "csv"

                print("filtering " + file + "...")
                os.makedirs(output_dir, exist_ok=True)
                filterData(file_in_path, file_out_path, input_format=format, keywords = keys,kw_field= kw_field, start_time= start, end_time= end,time_key = time_key, attrs=attrs, parse_quotes=quote)
                
            df = pd.read_csv(file_out_path)
            counts.append({'Ticker': keys[0], 'count': len(df)})

        counts_df = pd.DataFrame(counts)
        counts_df.to_csv(f"{output_dir}/tickerCounts.csv", index=False)

            # organiseFrame() #re-index the data in the csv by date  

        """
        Once we have filtered and extracted all the necessary data, we can start predicting the sentiment.
        This is done by using the model.py script, which uses the pretrained finBERT model. 
        """

        input_path = output_dir # path to the processed/ filtered data
        output_path = f"data/model_results/media-based/{dir}"
        overwrite = False

        start_time = time.time()
        m.predictAll(input_path, output_path, input_format="csv",  save=True, overwrite=overwrite, time_key='date_utc', text_key='text')
        end_time = time.time()

        print("Total execution time: ", timedelta(seconds=end_time-start_time))

    """
    Again, we repeat the same process for the news-based data, located in data/raw/news-based.
    """
    news_path = "data/raw/news-based/"
    output_news_path = "data/processed/news-based"
    output_nmodel_path = "data/model_results/news-based"
    c = False

    if c and os.path.exists(output_news_path):
        print(f"Cleaning {output_news_path}...")
        clean([output_news_path, output_nmodel_path])

    keysdict = [
        #[time_key, kw_field]
        ['Time', 'Headlines', True], # mixed articles from bloomberg, reuters, guardian
        ['published_at', 'hed', True], # reuters articles
    ]

    for i, dir in enumerate(os.listdir(news_path)):
        input_dir = f"{news_path}{dir}" # path to the extracted data
        output_dir = f"{output_news_path}/{dir}" # path to output the filtered data
        time_key = keysdict[i][0] # the col name for the timestamp in the data
        kw_field = keysdict[i][1] # the field to filter by, e.g., title, self_text, etc.
        attrs = [time_key, kw_field] # the attributes to be extracted from the data
        quote = keysdict[i][2] # flag to parse quotes
        counts = []

        for keys in worddict:
            print()
            print(f"-----------------Extracting for {output_dir}/{keys[0]}-----------------")
            print()

            file_out_path = f"{output_dir}/{keys[0]}({datetime.strptime(start,'%Y-%m-%d').strftime('%d.%m.%Y')}--{datetime.strptime(end,'%Y-%m-%d').strftime('%d.%m.%Y')}).csv"

            if os.path.exists(file_out_path):
                print(f"{file_out_path} already exists, skipping...")
                continue
            
            for file in os.listdir(input_dir):
                file_in_path = f"{input_dir}/{file}"
                format = "json" # the format of the data, e.g., json or csv
                if file.endswith(".csv"):
                    format = "csv"

                print("filtering " + file + "...")
                os.makedirs(output_dir, exist_ok=True)
                filterData(file_in_path, file_out_path, input_format=format, keywords = keys,kw_field= kw_field, start_time= start, end_time= end,time_key = time_key, attrs=attrs, parse_quotes=quote)
            # organiseFrame() #re-index the data in the csv by date 
            df = pd.read_csv(file_out_path)
            counts.append({'Ticker': keys[0], 'count': len(df)})

        counts_df = pd.DataFrame(counts)
        counts_df.to_csv(f"{output_dir}/tickerCounts.csv", index=False) 

        """
        Fetch the predictions for the news-based data.
        """

        input_path = output_dir
        output_path = f"data/model_results/news-based/{dir}"
        overwrite = False

        start_time = time.time()
        m.predictAll(input_path, output_path, input_format="csv",  save=True, overwrite=overwrite, time_key='date_utc', text_key='text')
        end_time = time.time()

        print("Total execution time: ", timedelta(seconds=end_time-start_time))

    # """
    # Then, we need to fetch the closing price history for each ticker from
    # the list, where the time period is the same as the extracted/filtered data 
    # in the step above.
    # """

    tickers = [key[0] for key in worddict]
    period =f"{datetime.strptime(start, '%Y-%m-%d').strftime('%d.%m.%Y')}--{datetime.strptime(end, '%Y-%m-%d').strftime('%d.%m.%Y')}"
    output_path = "data/tickersHistory"
    overwrite=True

    df = getClose(tickers, start, end) # here, start time & end time are the same as defined above
    # saveFrame(df, "test", overwrite=overwrite, index = True)

    for ticker in tickers:
        percent_change = ((df['Close'][ticker] - df['Open'][ticker])/df['Open'][ticker])*100
        # df2 = pd.DataFrame({'Close':df['Close'][ticker], 'Open':df['Open'][ticker], 'Percent Change':percent_change})
        df2 = pd.DataFrame({
            'Close': df['Close'][ticker],
            'Open': df['Open'][ticker],
            'High': df['High'][ticker],
            'Low': df['Low'][ticker],
            'Adj Close': df['Adj Close'][ticker],
            'Volume': df['Volume'][ticker],
            'Percent Change': percent_change
        })
        saveFrame(df2, output_path, keys= [f"{ticker}({period})"], overwrite=overwrite, index = True)

    """
    After getting the predicted values for each ticker, we evaluate how well the these sentiment values can 'predict' 
    the closing price of the corresponding ticker. This is where the evaluation.py comes in. There's a range of ways the 
    evaluation can be done:
    
    """
    print()
    print("Evaluating data...")
    wallstreetbets_results = "data/model_results/media-based/extracted_wsb_dumps"
    financialtweets_path = "data/model_results/media-based/financial-tweets"
    cnncTheGuardian_path = "data/model_results/news-based/mixed articles"
    reuters_path = "data/model_results/news-based/reuters_articles"
    datasets = {
        wallstreetbets_results: {"start": "2016-03-02", "end": "2020-12-30"}, 
        financialtweets_path: {"start": "2018-07-01", "end": "2018-07-31"},
        cnncTheGuardian_path: {"start": "2017-12-22", "end": "2020-07-19"},
        reuters_path: {"start": "2016-03-12", "end": "2020-12-30"}
        }

    outputArticle_path = "data/evaluations/news-based"
    outputMedia_path = "data/evaluations/media-based"
    c = False
    
    if c and os.path.exists(outputArticle_path):
        print(f"Cleaning {output_news_path}...")
        clean(outputArticle_path)
    if c and os.path.exists(outputMedia_path):
        print(f"Cleaning {outputMedia_path}...")
        clean(outputMedia_path)

    # dates = {
    #     "wsb": {"start": "2016-03-02", "end": "2020-12-30"}, 
    #     "tweets": {"start": "2018-07-01", "end": "2018-07-31"}, 
    #     "cnncTheGuardian": {"start": "2017-12-22", "end": "2020-07-19"}, 
    #     "reuters": {"start": "2016-03-12", "end": "2020-12-30"}
    # }

    shortStrategy = [2, 7, 15, 30]
    longStrategy = [90, 180, 365]
    strategies = [shortStrategy, longStrategy]
    # Q1 (First Quarter): January 1st to March 31st
    # Q2 (Second Quarter): April 1st to June 30th
    # Q3 (Third Quarter): July 1st to September 30th
    # Q4 (Fourth Quarter): October 1st to December 31st

    for i in range(2):
        if i == 0:
            length = "Short(1 month)"
        else:
            length = "Long(1 year)"
        
        for interval in strategies[i]:
            samepleInterval = interval
            for dataset, dates in datasets.items():
                if "media-based" in dataset and "wsb" in dataset:
                    outXlsx_path = f"{outputMedia_path}/{length}/{samepleInterval} interval wsb dataset.xlsx"
                elif "media-based" in dataset and "tweets" in dataset:
                    outXlsx_path = f"{outputMedia_path}/{length}/{samepleInterval} interval twitter dataset.xlsx"
                elif "news-based" in dataset and "mixed" in dataset:
                    outXlsx_path = f"{outputArticle_path}/{length}/{samepleInterval} interval cnbc+guardian dataset.xlsx"
                elif "news-based" in dataset and "reuters" in dataset:
                    outXlsx_path = f"{outputArticle_path}/{length}/{samepleInterval} interval reuters dataset.xlsx"

                if not os.path.exists(f"{outputMedia_path}/{length}"):
                    os.makedirs(f"{outputMedia_path}/{length}", exist_ok=True)
                if not os.path.exists(f"{outputArticle_path}/{length}"):
                    os.makedirs(f"{outputArticle_path}/{length}", exist_ok=True)
                
                with pd.ExcelWriter(outXlsx_path, engine = "xlsxwriter") as writer:
                    workbook = writer.book
                    
                    startrow = 2
                    startcol = 0

            
                    for ticker in os.listdir(dataset):
                        name = ticker.split('(')[0]
                        modelResult_df = loadFrame(f"{dataset}/{ticker}", 'csv')[0]
                        priceHistory_df = loadFrame(f"data/tickersHistory/{ticker}", 'csv')[0]

                        start = datetime.strptime(dates['start'], "%Y-%m-%d")
                        end = datetime.strptime(dates['end'], "%Y-%m-%d")

                        e = Evaluate(start, end, modelResult_df, priceHistory_df, samepleInterval)
                        accTable = e.accuracy()
                        logitAccTable = e.accuracyOnLogits()
                        print(accTable)
                        print(logitAccTable)
                        metrics = {
                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Total lines'],
                            'LabelsC': [
                                e.sumConfusionMatrix(accTable, 'accuracy'),
                                e.sumConfusionMatrix(accTable, 'precision'),
                                e.sumConfusionMatrix(accTable, 'recall'),
                                e.sumConfusionMatrix(accTable, 'f1'),
                                e.totalAcc
                            ],
                            'RawC': [
                                e.sumConfusionMatrix(logitAccTable, 'accuracy'),
                                e.sumConfusionMatrix(logitAccTable, 'precision'),
                                e.sumConfusionMatrix(logitAccTable, 'recall'),
                                e.sumConfusionMatrix(logitAccTable, 'f1'),
                                e.totalAccLogits
                            ]
                        }
                        metrics_df = pd.DataFrame(metrics)
                        
                        # Write the DataFrame to Excel, starting below the title
                        metrics_df.to_excel(writer, sheet_name='Sheet1',startcol=startcol, startrow=startrow, index=False)
                        worksheet = writer.sheets['Sheet1']

                        format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'font_size': 14})
                        worksheet.write(f'{xl_col_to_name(startcol)}{1}', name, format)

                        startcol += len(metrics_df.columns)
                        # ticker_df = pd.DataFrame({metrics_df.columns[0]: [name]}, index=[startrow])
                        # combinedShort_df = pd.concat([ticker_df, metrics_df], ignore_index=True)

                        # combinedShort_df.to_excel(writer, index=False, startrow=startrow, startcol=startcol, header=False)
                        # startcol += len(metrics_df.columns)

            # metrics_df.to_csv(f"{outputMedia_path}/{name} {samepleInterval} {start.strftime('%d.%m.%Y')}--{end.strftime('%d.%m.%Y')}.csv", index=False)
            # metrics_df.to_excel(f"{outputMedia_path}/{name} {samepleInterval} {start.strftime('%d.%m.%Y')}--{end.strftime('%d.%m.%Y')}.xlsx", index=False)
            # metrics_df.to_csv(f"{outCsv_path}/{name} {samepleInterval}.csv", index=False)
            # metrics_df.to_excel(f"{outXlsx_path}/{name} {samepleInterval}.xlsx", index=False)

if __name__ == "__main__":
    setCores(6)
    main()