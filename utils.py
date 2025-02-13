# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import codecs
import io
import re
import json
import pickle
import numpy as np
from config import OPENAI_API_KEY


class myOpenAI():
    def __init__(self):    
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_name = "text-embedding-3-large"

    def get_embedding(self, text):
       #text = text.replace("\n", " ") #??
       return self.client.embeddings.create(input = [text], model=self.embedding_name).data[0].embedding


def file_put_contents(filename, st):
    file = codecs.open(filename, "w", "utf-8")
    file.write(st)
    file.close()

def file_get_contents(name):
    f = io.open(name, mode="r", encoding="utf-8") #utf-8 | Windows-1252
    return f.read()

def pickle_save(path, obj):
    with open(path, "wb") as file: pickle.dump(obj, file)

def pickle_load(path):
    with open(path, "rb") as file: 
        obj = pickle.load(file)
        return obj

