''''

This script takes a flat file, reads it and displays the contents

'''

import pandas as pd
import sys

print("Libraries Successfully Imported")

def read_file(file_path):
    
    try:
        #using pandas function to read the file
        dataset=pd.read_csv(file_path)
        print(dataset.head())
        
    except FileNotFoundError:
        #handle error when the path given does not exist
        print("File Not Found, System exiting")
        sys.exit(1)
        
    except Exception as e:
        
        print("A Fatal error occured, System exiting")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    file_path = input("Enter the file path: ")
    read_file(file_path)

        
    
        
    