import pandas as pd
import os
import time
from tqdm import tqdm
from utils import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import timedelta
import torch

class model:
    """
    The implementation for the model used to predict the sentiment.
    """
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check if the GPU can be used, if not use cpu

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert") 
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    # model = torch.nn.DataParallel(model) 

    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    # result = pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score'])

    batch_size = 1000 ## default batch size;; adjust accordingly if you wish to boost/reduce performance load

    def predictAll(self, dir, output_path = "data/results", input_format = "json", save = False, overwrite = False, time_key = None, text_key = None):
        """
        Feed the parsed/filtered data from all 4 datasets of both data type through the model
        dir: str
          path to the directory containing the data
        input_format: str
            format of the input data
        """
        d = os.listdir(dir)

        for file in d:
            if file == "tickerCounts.csv":
                continue
            print("Procesing file: ", f"{dir}/{file}", "...")
            result = self.predict(f"{dir}/{file}", f"{output_path}/{file}", input_format,overwrite=overwrite, save = save, time_key = time_key, text_key = text_key)
            print()
            if result is None:
                print(f"{output_path}/{file}"," already processed, skipping...")
                continue

    def predict(self, input_path, output_path, input_format = "json", overwrite = False, save = False, time_key = None, text_key = None):
        if os.path.exists(output_path) and overwrite == False:
            #data alreqady processed and saved, and overwrite flag not set
            return
        
        df = loadFrame(input_path, format=input_format)

        if df[1] == True:
            chunk_iterator = df[0]
            print("Setting up chunk iterator of size: ", chunk_iterator.chunksize)

            with open(output_path, 'a', newline='', encoding='utf-8') as output_file:
                write_header = True

                for chunk in tqdm(chunk_iterator, desc="Processing chunks"):
                    dates= chunk[time_key].to_list()
                    titles = chunk[text_key].to_list()

                    start_time = time.time()
                    chunk_result = self.run(dates, titles)
                    end_time = time.time()

                    print("Chunk took: ", timedelta(seconds=end_time-start_time), end="", flush=True)

                    chunk_result['Prediction'] = chunk_result.Prediction.apply(lambda x: self.label_dict[x])

                    chunk_result.to_csv(output_file, mode='a', header=write_header, encoding='utf-8')
                    write_header = False

        else:
            df = df[0]
            dates = df[time_key].to_list()
            titles = df[text_key].to_list()

            start_time = time.time()
            result = self.run(dates, titles)
            end_time = time.time()

            print("Time taken: ", timedelta(seconds=end_time-start_time), end="", flush=True)

            result['Prediction'] = result.Prediction.apply(lambda x: self.label_dict[x])

            if save:
                saveFrame(result, output_path, overwrite= overwrite, full_path = True)
        return result
    
    def run(self, dates, data, batch_size=batch_size):
        """
        Adaptation from a part of FinBERT source code, full implementation can be found on the ProsusAI/finbert repo on GitHub: https://github.com/ProsusAI/finBERT
        """
        no_batches = len(data)/batch_size
        # dates= chunks(dates, batch_size)
    
        data_chunks = chunks(data, batch_size)
        date_chunks = chunks(dates, batch_size)
        # for i, batch in enumerate(chunks(data, batch_size), start=1):
        result = pd.DataFrame(columns=['Date', 'Text', 'Logits', 'Prediction', 'Sentiment_score'])

        for i, (data_batch, date_batch) in enumerate(zip(data_chunks, date_chunks), start=1):
        
            print(f"\rProcessing batch {i} out of {no_batches} ", end="", flush=True)
            examples = [InputExample(str(i), sentence) for i, sentence in enumerate(data_batch)] # create examples from the data batch

            features = convert_examples_to_features(examples, self.label_list, 64, self.tokenizer) # convert the examples to features

            """ Convert the features to tensors"""
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device_cpu) 
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(self.device_cpu)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(self.device_cpu)

            with torch.no_grad(): # temporarily disable gradient computation during inference to reduce memory consumption (reccommended to use when not training the model)
                """
                For each batch, feed the input ids, attention mask and token type ids to the model and get the logits.
                Then normalise the logits using the softmax function and concatenate the batch results to all results dataframe.
                """
                self.model.to(self.device_cpu)

                logits = self.model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
                logits = softmax(np.array(logits.cpu()))
                sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
                predictions = np.squeeze(np.argmax(logits, axis=1))

                batch_result = {'Date': date_batch,
                                'Text': data_batch,
                                'Logits': list(logits),
                                'Prediction': predictions,
                                'Sentiment_score': sentiment_score}

            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result, batch_result], ignore_index=True)
            
        return result


if __name__ == "__main__":
    """
    Example usage of the model class
    """
    m = model()
    psutil.Process(os.getpid()).cpu_affinity(m.setCores(7))
    setCores(7)
    print(torch.__version__)
    print("Cuda status", torch.cuda.is_available())
    # predict("unprocessed data\extracted_wsb_dumps\wallstreetbets__submissions", "wsb_submissions_csv", keys = ["Apple"], save=True, overwrite=True)
    m.predictAll("unprocessed data/extracted_wsb_dumps", save=True)


    


