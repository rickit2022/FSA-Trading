from utils import loadFrame, saveFrame
import pandas as pd

class Evaluate:
    """
    Evaluate the accuracy, precision, recall, f1-score of the model's predictions by
    comparing the model's sentiment predictions with the actual price changes for 
    both data types. For each metrics, the 2 methods of calculations are used. 
    This will result in 4 metrics x 2 calculations = 8 results for each data type.

    """
    start = None
    end = None
    modelResult_df = None
    priceHistory_df = None
    sampleRange = None
    periods = None
    accTable = None
    totalAcc = 0
    totalAccLogits = 0
    # explainer = shap.Explainer(self.model)

    def __init__(self, start, end, modelResult_df, priceHistory_df, sampleRange = None):
        """
        Time range: consists of start and end dates
        modelResult_df: DataFrame containing the model's predictions
        priceHistory_df: DataFrame containing the historical price data
        sampleRange: the range of days to sample the data
        periods: divide the time range into periods of sampleRange
        """
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.sampleRange = sampleRange
        self.modelResult_df = modelResult_df
        self.modelResult_df['Date'] = pd.to_datetime(self.modelResult_df['Date'])
        self.priceHistory_df = priceHistory_df
        self.priceHistory_df['Date'] = pd.to_datetime(self.priceHistory_df['Date'])

        self.periods = [pd.to_datetime(start), pd.to_datetime(end)]
        if sampleRange is not None:
            self.periods = pd.date_range(start, end, freq=f'{sampleRange}D')
            if self.periods[-1] < self.end:
                # make the last period the end date if the last period is less than the end date
                self.periods = self.periods.append(pd.DatetimeIndex([end]))

    def accuracy(self):
        """
        The LabelsC classification method, taking the highest count of sentiment predictions
        """
        accuracyTable = []
        modelResult_df = self.modelResult_df
        priceHistory_df = self.priceHistory_df

        for i in range(0, len(self.periods) - 1):
            period_start = self.periods[i]
            period_end = self.periods[i + 1]
            
            filtered_df = modelResult_df[(modelResult_df['Date'] >= period_start) & (modelResult_df['Date'] < period_end)]
            counts = filtered_df['Prediction'].value_counts()

            if 'neutral' in counts:
                counts = counts.drop('neutral')
                #neutral doesn't provide much value here, so we discard all neutral sentiments
            total = counts.sum()
            self.totalAcc += total

            highestSentiment = None
            highestCount = 0
            if not counts.empty:
                highestSentiment = counts.idxmax()
                highestCount = counts.max()
            
            closeStart = priceHistory_df[priceHistory_df['Date'] == period_start]['Close']
            closeEnd = priceHistory_df[priceHistory_df['Date'] == period_end]['Close']
            """
            There are gaps in the data, potentially due to the market being closed.
            We will shift the period_start and period_end dates to find the closest date where 
            thre is data.
            """
            maxShift = 10 # max number of days to shift. If no data is found after maxShift days, return None for the calculation
            daysLooked = 0
            while closeStart.empty and daysLooked <= maxShift:
                period_start += pd.Timedelta(days=1)
                closeStart = priceHistory_df[priceHistory_df['Date'] == period_start]['Close']
                daysLooked += 1
            while closeEnd.empty and daysLooked <= maxShift:
                period_end += pd.Timedelta(days=1)
                closeEnd = priceHistory_df[priceHistory_df['Date'] == period_end]['Close']
                daysLooked += 1

            closeStart = closeStart.iloc[0] if not closeStart.empty else None
            closeEnd = closeEnd.iloc[0] if not closeEnd.empty else None

            change = 0
            pChange = 0

            if closeStart and closeEnd:
                change = round(closeEnd - closeStart, 2) 
                pChange = round(((closeEnd - closeStart) / closeStart) * 100, 2) # percentage change

            result = self.confusionMatrixFeatures(highestSentiment, pChange)

            accuracyTable.append({
                'Date': f"{period_start.strftime('%d/%m/%Y')}-{period_end.strftime('%d/%m/%Y')}",
                'No. sentiments': total,
                'Predicted trend': highestSentiment,
                'HighestCount': highestCount,
                'Change': change,
                'PChange': pChange,
                'Result': result
            })
        return pd.DataFrame(accuracyTable)
        
    def confusionMatrixFeatures(self, highestSentiment, pChange):
        """
        generate the confusion matrix features for the LabelsC method
        """
        if highestSentiment== 'negative' and pChange < 0:
            #True negative
            return "TN"
        elif highestSentiment == 'positive' and pChange > 0:
            #True positive
            return "TP"
        elif highestSentiment == 'positive' and pChange < 0:
            #False positive
            return "FP"
        elif highestSentiment == 'negative' and pChange > 0:
            #False negative
            return "FN"
        else:
            return None
        
    def confusionMatrixFeaturesLogits(self, logitScore, pChange):
        """
        generate the confusion matrix features for the RawC method
        """
        if logitScore > 0 and pChange > 0:
            return "TP"
        elif logitScore < 0 and pChange < 0:
            return "TN"
        elif logitScore > 0 and pChange < 0:
            return "FP"
        elif logitScore < 0 and pChange > 0:
            return "FN"
        else:
            return None
        
    def accuracyOnLogits(self):
        """
        The RawC classification method, taking the mean of the sentiment scores (difference in negative to positive logits)
        """
        accuracyTable = []
        modelResult_df = self.modelResult_df
        priceHistory_df = self.priceHistory_df

        for i in range(0, len(self.periods) - 1):
            period_start = self.periods[i]
            period_end = self.periods[i + 1]
            
            filtered_df = modelResult_df[(modelResult_df['Date'] >= period_start) & (modelResult_df['Date'] < period_end)]
            neutralSentiments = filtered_df[filtered_df['Prediction'] == 'neutral']['Prediction'].count()
            filtered_df = filtered_df[~(filtered_df['Prediction'] == 'neutral')]

            meanRaw = filtered_df['Sentiment_score'].mean()
            posSentiments = filtered_df[filtered_df['Prediction'] == 'positive']['Prediction'].count()
            negSentiments = filtered_df[filtered_df['Prediction'] == 'negative']['Prediction'].count()
            total = filtered_df['Prediction'].count()
            self.totalAccLogits += total
            
            closeStart = priceHistory_df[priceHistory_df['Date'] == period_start]['Close']
            closeEnd = priceHistory_df[priceHistory_df['Date'] == period_end]['Close']

            #same procedure follow as the LabelsC method (normal accuracy)
            maxShift = 10 # max number of days to shift if no data is found
            daysLooked = 0
            
            while closeStart.empty and daysLooked <= maxShift:
                period_start += pd.Timedelta(days=1)
                closeStart = priceHistory_df[priceHistory_df['Date'] == period_start]['Close']
                daysLooked += 1
            while closeEnd.empty and daysLooked <= maxShift:
                period_end += pd.Timedelta(days=1)
                closeEnd = priceHistory_df[priceHistory_df['Date'] == period_end]['Close']
                daysLooked += 1

            closeStart = closeStart.iloc[0] if not closeStart.empty else None
            closeEnd = closeEnd.iloc[0] if not closeEnd.empty else None

            change = 0
            pChange = 0

            if closeStart and closeEnd:
                # some days don't have info, may be due to the market being closed
                change = round(closeEnd - closeStart, 2)
                pChange = round(((closeEnd - closeStart) / closeStart) * 100, 2)
            
            result = self.confusionMatrixFeaturesLogits(meanRaw, pChange)

            accuracyTable.append({
                'Date': f"{period_start.strftime('%d/%m/%Y')}-{period_end.strftime('%d/%m/%Y')}",
                'No. sentiments': total,
                'MeanRawScore': meanRaw,
                'Positive': posSentiments,
                'Negative': negSentiments,
                'Neutral': neutralSentiments,
                'Change': change,
                'PChange': pChange,
                'Result': result
            })
        return pd.DataFrame(accuracyTable)

    def sumConfusionMatrix(self, table, type):
        """
        Sum the confusion matrix features to calculate the metrics, then 
        calculate the metrics asked.
        """
        tp = table[table['Result'] == 'TP']['Result'].count()
        tn = table[table['Result'] == 'TN']['Result'].count()
        fp = table[table['Result'] == 'FP']['Result'].count()
        fn = table[table['Result'] == 'FN']['Result'].count()
        print(" tp: ", tp, " tn:", tn, " fp: ", fp, " fn: ", fn)

        if type == 'precision':
            if (tp + fp) == 0:
                return None
            else:
                return round(tp / (tp + fp), 3)
        elif type == 'recall':
            if (tp + fn) == 0:
                return None
            else:
                return round(tp / (tp + fn), 3)
        elif type == 'f1':
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
            
            f1_score = 0
            if (precision + recall) > 0:
                f1_score = 2 * ((precision * recall) / (precision + recall))
                
            return round(f1_score, 3)
        elif type == 'accuracy':
            return round((tp + tn) / (tp + tn + fp + fn), 3)
    

if __name__ == "__main__":
    """Example usage of the Evaluate class."""
    # start = '2017-05-1'
    # end = '2017-05-7'
    start= "2016-03-02"
    end= "2020-12-30"
    samepleRange= 365 # 7,14,30,60, 150, 365

    modelResult_df = loadFrame("data/model_results/media-based/extracted_wsb_dumps/AAPL(02.03.2016--30.12.2020).csv", 'csv')[0]
    priceHistory_df = loadFrame('data/tickersHistory/AAPL(02.03.2016--30.12.2020).csv', 'csv')[0]     

    e = Evaluate(start, end, modelResult_df, priceHistory_df, samepleRange)

    accTable = e.accuracy()
    logitAccTable = e.accuracyOnLogits()
    print("Label classification:", accTable)
    print("Normalised scores classifcation:", logitAccTable)
    
    # print("Accuracy using mean raw:", e.sumConfusionMatrix(logitAccTable, 'accuracy'))
    # print("Precision using binary polarity:", e.sumConfusionMatrix(accTable, 'precision'))
    # print("Recall using binary polarity:", e.sumConfusionMatrix(accTable, 'recall'))
    # print("F1 using binary polarity:", e.sumConfusionMatrix(accTable, 'f1'))

    # print("Accuracy using binary polarity:", e.sumConfusionMatrix(accTable, 'accuracy'))
    # print("Precision using mean raw:", e.sumConfusionMatrix(logitAccTable, 'precision'))
    # print("Recall using mean raw:", e.sumConfusionMatrix(logitAccTable, 'recall'))
    # print("F1 using mean raw:", e.sumConfusionMatrix(logitAccTable, 'f1'))

    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Lines'],
        'Labels Classification': [
            e.sumConfusionMatrix(accTable, 'accuracy'),
            e.sumConfusionMatrix(accTable, 'precision'),
            e.sumConfusionMatrix(accTable, 'recall'),
            e.sumConfusionMatrix(accTable, 'f1'),
            e.totalAcc
        ],
        'Normalised scores Classification': [
            e.sumConfusionMatrix(logitAccTable, 'accuracy'),
            e.sumConfusionMatrix(logitAccTable, 'precision'),
            e.sumConfusionMatrix(logitAccTable, 'recall'),
            e.sumConfusionMatrix(logitAccTable, 'f1'),
            e.totalAccLogits
        ]
    }
    pd.set_option('display.precision', 3)
    metrics_df = pd.DataFrame(metrics)

    print(metrics_df)