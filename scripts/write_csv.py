'''
This script writes data to a csv file 
It creates fake data from the python library faker

'''

import pandas as pd
import sys 
import os
from faker import Faker
from datetime import datetime


print("Libraries Successfully Imported")

fake = Faker()
#we create a genre generator for our movies
def genre_generator():
    return fake.random_element(elements=('Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'))

#we create a language generator for our movies

def language_generator():
    return fake.random_element(elements=('English', 'French', 'Spanish', 'German', 'Italian', 'Portuguese', 'Japanese', 'Chinese', 'Hindi'))

#we create a duration generator for our movies

def duration_generator():
    return fake.random_int(min=60, max=200)


#we create a rankings generator for our movies

def rating_generator():
    return fake.random_int(min=1, max=10)

def join_words(words):
    return ' '.join(words)


def gen_data():
    movie_names=' '.join(fake.words())
    
    date=datetime.strftime(fake.date_time_this_decade(), '%Y-%m-%d')
    movie_genre=genre_generator()
    movie_language=language_generator()
    movie_duration=duration_generator()
    movie_rating=rating_generator()
    
    return movie_names, date, movie_genre, movie_language, movie_duration, movie_rating
    

def write_file(filename):
    file_path = os.path.join('../data', filename)
    print("Your file path is: {} ".format(file_path))
    
    try:
        
        with open(file_path, 'w') as f:
            f.write("Movie Name,Release Date,Genre,Language,Duration,Rating\n")
            for _ in range(1000):
                movie_names, date, movie_genre, movie_language, movie_duration, movie_rating = gen_data()
                f.write("{},{},{},{},{},{}\n".format(movie_names, date, movie_genre, movie_language, movie_duration, movie_rating))
    
    
    except Exception as e:
        
        print("A Fatal error occured, System exiting")
        sys.exit(1)
        
        
        
if __name__ == "__main__":
    filename = input("Enter the file name: ")
    write_file(filename)
        
    