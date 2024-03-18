class DataPreprocessor:
    def __init__(self, mima_col, cat_col):
        self.numerical_features = mima_col
        self.categorical_features = cat_col


        # Combine transformers using ColumnTransformer
        cat_pipe=Pipeline([('mode',simo),('ohe',ohe)])
        num_pipe=Pipeline([('mean',sim),('minmax',mima)])
        ct=ColumnTransformer([('cat',cat_pipe,self.categorical_features),('num',num_pipe,self.numerical_features)])

        #final_pipe=Pipeline([('ct',ct)])
        self.preprocessor = ct


    def fit(self,data):
        # Fit the preprocessing transformers
        self.preprocessor.fit(data)

    def transform(self, sdata):
        # Apply preprocessing and scaling
        transformed_data = self.preprocessor.transform(sdata)

        return transformed_data