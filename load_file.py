import pandas as pd

def load_file():
    books_data = pd.read_csv("file4.csv", encoding = 'utf-8')

    return books_data