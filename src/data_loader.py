from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm


class DataLoader(ABC):
    """
      Provide DataLoader interface for documentation purposes.
      Used for CSVDataLoader and DBDataLoader.
      
        params
        -------
        X : the dataframe being treated


        returns
        -------
        X : new dataframe with treated outlier values        
    """      

    @abstractmethod
    def combine_files():
        pass

    @abstractmethod
    def load(self):
        pass


class CSVDataLoader(DataLoader):
    """
      Implements load() method with pandas read_csv functionality

        
        params
        -------
        X : the dataframe being treated


        returns
        -------
        X : new dataframe with treated outlier values        
    """

    def combine_files(self, data_folder_path):
        import glob

        concatenated = pd.DataFrame()

        # Define the file extension
        file_pattern = 'graduation*.csv'
        print(file_pattern)

        # Extract the list of files with the extension
        file_list = glob.glob(file_pattern, root_dir=data_folder_path)

        # Print the list of files
        print(file_list)

        DFs = []

        for file in file_list:
            print(f'reading {file} ... ')
            df = CSVDataLoader().load(data_folder_path / file)
            # display(df.head(2))
            DFs.append(df)
            print()

        data = pd.concat(DFs)
        print(f'{data.shape = }')
        # display(data.head(2))

        output_file = data_folder_path / 'graduation_rate.csv'
        print(output_file)
        if output_file.exists():
            print(f'{output_file} exists, not overwritten.')
        else:
            data.to_csv(output_file, index=False, header=True, mode='w')

    def load(self, file_path):
        print('Loading CSV data...')

        # Define the chunk size
        chunk_size = 500

        # Define the total number of rows in the file
        total_rows = sum(1 for line in open(file_path))

        # Initialize an empty dataframe to store the data
        df = pd.DataFrame()

        # Loop through the file in chunks and append to the dataframe
        for chunk in tqdm(pd.read_csv(
                file_path, chunksize=chunk_size),
                total=total_rows//chunk_size):
            df = pd.concat([df, chunk], ignore_index=True)

        print('Finished processing the CSV file.')
        print(f'{df.shape = }')
        return df

    def selective_load(self, file_path, cols_to_use):
        print('Loading CSV data...')
        return pd.read_csv(file_path, usecols=cols_to_use)

    def data_profile(self):
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            self.to_pandas(), title="Time-Series EDA full df")
        # profile.to_notebook_iframe()
        profile.to_file("/artifacts/reports/full_df_ProfileReport.html")

    def save(self, df, file_path):
        print('Saving CSV data...')

        print(file_path)
        if file_path.exists():
            print(f'{file_path} exists, not overwritten.')
        else:
            df.to_csv(file_path, index=False, header=True, mode='w')


class DBDataLoader(DataLoader):
    '''
      OOP-styled code for loading dataset from database.
    '''
    pass
