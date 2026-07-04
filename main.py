from sentence_transformers import SentenceTransformer
# it is a model that basically converts text -> embedding(numerical)
from tqdm.auto import tqdm
#for progress bar
import pandas as pd
import numpy as np
import torch
# for ml models
from time import sleep
#sleep is basically used for adding animations or maybe waiting for some time
from services.retrieve import print_wrapped,retrieve_relevant_resources,print_top_results_and_scores
# retrieve_relevant_resources -> searches relevant data
from llm.get_response import ask
# to get answers from llm model like gemini

device = "cpu"
#running our code on cpu
query = input("Enter your query: ")



print("The device is set to:",device)
print("This may take a while...")

sleep(2)


print("Loading the Saved Embeddings DataFrame...")

embeddings_df_save_path = "data/text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df = pd.read_csv(embeddings_df_save_path)
# here basically the already stored embedded file is taken -> loading takeing place



print("Converting the 'embedding' column to a numpy array...")
sleep(2)
text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
# converting embedded file (string -> array)

pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient="records")
# making every row a dictionary
embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()), dtype=torch.float32).to(device)
# converting embedded format -> torch tensor(ML friendly)
print("Successsfully Converted the 'embedding' column to a torch tensor.")
sleep(2)

print("\n\n")
print("Retrieving the most relevant resources...")
print("\n\n")

sleep(2)


print_top_results_and_scores(query=query, embeddings=embeddings, pages_and_chunks=pages_and_chunks)

 
print("Using Gemini to generate a response...")
print("\n\n")
sleep(2)


# we will make query also in embedding format , we will get relevant data , LLM will answer 

ans = ask(query=query, embeddings=embeddings, pages_and_chunks=pages_and_chunks,embeddings_df_save_path=embeddings_df_save_path)

print_wrapped(ans)









