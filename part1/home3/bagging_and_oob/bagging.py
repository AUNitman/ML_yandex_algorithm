import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for _ in range(self.num_bags):
            indices = np.random.choice(data_length, data_length, replace=True)  # Fixed: choose indices, not data values
            self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        '''
        self.data = data  # Store data if needed for OOB
        self.target = target  # Store target if needed for OOB
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag = data[self.indices_list[bag]]  # Fixed: use indices to get data samples
            target_bag = target[self.indices_list[bag]]  # Fixed: use indices to get target samples
            self.models_list.append(model.fit(data_bag, target_bag))
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = np.array([model.predict(data) for model in self.models_list])  # Fixed: models_list not model_list
        return np.mean(predictions, axis=0)  # Fixed: average along the correct axis
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for i, model in enumerate(self.models_list):
            # Create mask for OOB samples (not in the bag)
            oob_mask = ~np.isin(np.arange(len(self.data)), self.indices_list[i])
            oob_data = self.data[oob_mask]
            
            if len(oob_data) > 0:
                oob_pred = model.predict(oob_data)
                # Store predictions for each OOB sample
                for j, pred in zip(np.where(oob_mask)[0], oob_pred):
                    list_of_predictions_lists[j].append(pred)
        
        self.list_of_predictions_lists = list_of_predictions_lists  # Doesn't need to be numpy array
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([np.mean(preds) if preds else np.nan for preds in self.list_of_predictions_lists])
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        if not self.oob:
            raise ValueError("OOB score is only available when oob=True")
            
        self._get_averaged_oob_predictions()
        valid_mask = ~np.isnan(self.oob_predictions)  # None becomes nan in numpy array
        if np.sum(valid_mask) == 0:
            return None
        return np.mean((self.target[valid_mask] - self.oob_predictions[valid_mask]) ** 2)