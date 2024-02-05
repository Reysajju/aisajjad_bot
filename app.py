import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load the initial dataset
initial_data = pd.DataFrame({'text': ['Hello', 'Hi', 'How are you?', 'Goodbye'],
                              'label': ['bot', 'bot', 'bot', 'bot']})

# Function to train the model and save it
def train_and_save_model(data):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(data['text'], data['label'])
    return model

def train_and_save_model(data):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    # Drop rows with NaN values in 'text' column
    data.dropna(subset=['text'], inplace=True)
    
    model.fit(data['text'], data['label'])
    return model

# Function to interact with the user
def chatbot():
    dataset_filename = '/content/datasets.csv'
    model = train_and_save_model(pd.read_csv(dataset_filename))  # Load model from existing dataset

    while True:
        user_input = input("You: ")
        new_data = pd.DataFrame({'text': [user_input], 'label': ['user']})
        update_and_save_dataset(dataset_filename, new_data)

        response = model.predict([user_input])[0]
        print(f"Bot: {response}")

        new_data['label'] = 'bot'
        update_and_save_dataset(dataset_filename, new_data)
        model = train_and_save_model(pd.read_csv(dataset_filename))

# Start the chatbot
chatbot()
