from KMeanServices import KMeansService
import pandas as pd

class MainApplication:
    def __init__(self, file_path):
        self.file_path = file_path
        self.service = KMeansService()

    def upload_file(self):
        df = pd.read_csv(self.file_path)
        return df

    def process_file(self):
        df = self.upload_file()
        df_numerical = df[['Species', 'PFG']]
        df_clustered = self.service.fit_predict(df_numerical)
        self.service.plot_clusters(df_clustered)

if __name__ == "__main__":
    file_path = 'data\PFG.csv' 
    app = MainApplication(file_path)
    app.process_file()
