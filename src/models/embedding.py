import sys
import time
from openai import OpenAI
from typing import Union, List
import random
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

class Embedding_API:
    def __init__(self, model_name="embedding-3", api_key=None, base_url=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    def respond(self, str_list):

        repeat_num = 0
        response_data = None
        while response_data == None:
            repeat_num += 1
            if repeat_num>5:
                response_data = "I Don't Know!"
            try:
                completion = self.client.embeddings.create(
                    model=self.model,
                    input = str_list
                )
                embedding_list = [data.embedding for data in completion.data]
                break
            except KeyboardInterrupt:
                sys.exit()
            except:
                print("Request timed out, retrying...")
                
        return embedding_list
    def compute_similarity(self, X: List[str], Y: List[str]):
        
        phi_X = self.respond(X)
        phi_Y = self.respond(Y)
        phi_X  = F.normalize(torch.tensor(phi_X), p=2, dim=1)
        phi_Y  = F.normalize(torch.tensor(phi_Y), p=2, dim=1)
        sim = torch.diagonal(util.pytorch_cos_sim(phi_X, phi_Y), 0)
        
        return sim.numpy().tolist()