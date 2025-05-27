"""
SMALL DESCRIPTION
"""

import torch
from collections.abc import Iterator
from sklearn.preprocessing import StandardScaler

class Evaluation:
    def __init__(self, 
                 model,
                 dataset,
                 concept_name = 'concept:name',
                 eos_value = 'EOS',
                 growing_num_values = ['case_elapsed_time'],
                 all_cat = None,
                 all_num = None):
        
        self.model = model
        self.dataset = dataset

        self.concept_name = concept_name
        self.concept_name_id = [i for i, cat in
                                enumerate(self.dataset.all_categories[0])
                                if cat[0] == self.concept_name][0]
        self.eos_value = eos_value
        self.eos_id = [v for k, v in
                       self.dataset.all_categories[0][self.concept_name_id][2].items()
                       if k == self.eos_value][0]
        
        self.growing_num_values = growing_num_values

        self.prefix_cat_attributes = self.model.enc_feat[0]
        self.prefix_num_attributes = self.model.enc_feat[1]
        prefix_categories = [ cat_tuple for cat_tuple in self.dataset.all_categories[0] if cat_tuple[0] in self.prefix_cat_attributes]
        self.inverted_prefix_categories = [{v: k for k, v in s[2].items()} for s in prefix_categories]

        self.all_cat_attributes = all_cat if all_cat else [cat[0] for cat in self.dataset.all_categories[0]]
        self.all_num_attributes = all_num if all_num else [num[0] for num in self.dataset.all_categories[1]]
        suffix_categories = [ cat_tuple for cat_tuple in self.dataset.all_categories[0] if cat_tuple[0] in self.all_cat_attributes]
        self.inverted_suffix_categories = [{v: k for k, v in s[2].items()} for s in suffix_categories]
        
        self.cases = self._get_cases_from_dataset()

    def _get_cases_from_dataset(self):
        cases = {}
        for event in self.dataset:
            suffix = event[0][self.concept_name_id][-self.dataset.encoder_decoder.min_suffix_size:]
            if torch.all(suffix  == self.eos_id).item():
                cases[event[2]] = event
        return cases
    
    def _iterate_case(self, case : tuple[list[torch.Tensor], list[torch.Tensor]]) -> Iterator[tuple]:
        current_prefix = ([torch.zeros_like(cat_attribute).unsqueeze(0) for cat_attribute in case[0]],
                        [torch.zeros_like(num_attribute).unsqueeze(0) for num_attribute in case[1]])
        
        current_suffix = ([torch.clone(cat_attribute).unsqueeze(0) for cat_attribute in case[0]],
                        [torch.clone(num_attribute).unsqueeze(0) for num_attribute in case[1]])
        
        prefix_length = 0
        for i in range(case[0][0].shape[0] - self.dataset.encoder_decoder.min_suffix_size- 1):
            for j in range(len(current_prefix[0])):
                current_prefix[0][j][0] = torch.roll(current_prefix[0][j][0], -1)
                current_prefix[0][j][0, -1] = case[0][j][i]
                current_suffix[0][j][0] = torch.roll(current_suffix[0][j][0], -1)
                current_suffix[0][j][0, -1] = 0

            for j in range(len(current_prefix[1])):
                current_prefix[1][j][0] = torch.roll(current_prefix[1][j][0], -1)
                current_prefix[1][j][0, -1] = case[1][j][i]
                current_suffix[1][j][0] = torch.roll(current_suffix[1][j][0], -1)
                current_suffix[1][j][0, -1] = 0

            if prefix_length or case[0][self.concept_name_id][i]:
                prefix_length += 1
                yield prefix_length, current_prefix, current_suffix
                
    def _get_num_prediction_with_means(self, pred_mean, last_means):
        result = {}
        for c in self.all_num_attributes:
            if c in self.growing_num_values:
                result[c+'_mean'] = torch.max(pred_mean[c+'_mean'], last_means[c+'_mean'])
            else:
                result[c+'_mean'] = pred_mean[c+'_mean']
            
        return result
    
    # NEW: Ncessary?
    def _get_num_prediction_with_vars(self, pred_vars):
        result = {}
        for c in self.all_num_attributes:   
            result[c+'_var'] = pred_vars[c+'_var']
        return result

    def _disable_model_dropout(self, model):
        storage = (model.dropout,
                   model.encoder.first_layer.p_logit, [layer.p_logit for layer in model.encoder.hidden_layers],
                   model.decoder.first_layer.p_logit, [layer.p_logit for layer in model.decoder.hidden_layers])
        model.dropout = 0.0
        model.encoder.first_layer.p_logit = 0.0
        for hl in model.encoder.hidden_layers:
            hl.p_logit = 0.0
        model.decoder.first_layer.p_logit = 0.0
        for hl in model.decoder.hidden_layers:
            hl.p_logit = 0.0
        return storage
    
    def _enable_dropout(self, model, dropout_rates):
        model.dropout = dropout_rates[0]
        model.encoder.first_layer.p_logit = dropout_rates[1]
        for i, hl in enumerate(model.encoder.hidden_layers):
            hl.p_logit = dropout_rates[2][i]
        model.decoder.first_layer.p_logit = dropout_rates[3]
        for i, hl in enumerate(model.decoder.hidden_layers):
            hl.p_logit = dropout_rates[4][i]


    def _predict_suffix_with_means(self, prefix, prefix_len):
        # disable dropout
        dropout_rates = self._disable_model_dropout(self.model)
        self.model.decoder 
        
        # Prediction by model
        prediction, (h, c), z = self.model.inference(prefix=prefix)
        
        suffix = []
        max_iteration = self.dataset.encoder_decoder.window_size - self.dataset.encoder_decoder.min_suffix_size - prefix_len
        i = 0
        eos_predicted = lambda prediction : torch.argmax(prediction[0][0][self.concept_name+'_mean']) == self.eos_id
        
        last_means = {a+'_mean' : prefix[1][self.all_num_attributes.index(a)][:,-1].unsqueeze(1) for a in self.growing_num_values}
        
        while i <= max_iteration and not eos_predicted(prediction):
            cat_prediction = {k : torch.argmax(cat_pred, keepdim=True) for k, cat_pred in prediction[0][0].items()}
            
            """
            # If log_normal
            if isLogNormal:                       
                result = dict()
                # Categorical predictions
                for i, k in enumerate(cat_prediction.keys()):
                    attribute_name = k[:-5]  # clip the _mean            
                    if cat_prediction[k].item():
                        result[attribute_name] = self.inverted_suffix_categories[i][cat_prediction[k].item()]
                    else:
                        result[attribute_name] = None
                        
                num_prediction = self._get_num_prediction_with_means(prediction[0][1], last_means)
                
                # Numerical predictions
                num_prediction_vars = prediction[1][1]
                for i, k in enumerate(num_prediction.keys()):
                    attribute_name = k[:-5]  # clip the _mean
                    
                    attribute_value = num_prediction[k].item()
                    attribute_value_logvars = num_prediction_vars[attribute_name+'_var'].item()
                    attribute_value_vars = np.exp(attribute_value_logvars)
                    
                    scaler = self.dataset.encoder_decoder.continuous_encoders[attribute_name]
                    x_stand_log = np.array([[attribute_value, attribute_value_vars]])  # shape (1, 2)
                    x = scaler.inverse_transform(x_stand_log)
                    result[attribute_name] = x[0]
                    
                readable_prediction = result
                            
            # If normal
            else:
                num_prediction = self._get_num_prediction_with_means(prediction[0][1], last_means)
                readable_prediction = self.prediction_to_readable(cat_prediction, num_prediction)
            """
            
            num_prediction = self._get_num_prediction_with_means(prediction[0][1], last_means)
            
            readable_prediction = self.prediction_to_readable(cat_prediction, num_prediction)
            suffix.append(readable_prediction)
            last_means = {key: tensor.clone() for key, tensor in num_prediction.items()}
            
            prediction, (h, c) = self.model.inference(last_event=(list(cat_prediction.values()), list(num_prediction.values())), hx=(h,c), z=z)
            
            i += 1
        
        self._enable_dropout(self.model, dropout_rates)
        
        return suffix
    
    
    def prediction_to_readable(self, cat_prediction, num_prediction):
        result = dict()
        # Categorical predictions
        for i, k in enumerate(cat_prediction.keys()):
            attribute_name = k[:-5]  # clip the _mean            
            if cat_prediction[k].item():
                result[attribute_name] = self.inverted_suffix_categories[i][cat_prediction[k].item()]
            else:
                result[attribute_name] = None
        # Numerical predictions
        for i, k in enumerate(num_prediction.keys()):
            attribute_name = k[:-5]  # clip the _mean
            attribute_value = num_prediction[k].item()
            
            #if attribute_value > 5:
            #    print("Very Large Encoded Num Prediction: ", attribute_value)
            
            scaler = self.dataset.encoder_decoder.continuous_encoders[attribute_name]             
            result[attribute_name] = self.inverse_transform(scaler, attribute_value)
        return result
    
    def inverse_transform(self, scaler, x_scaled):
        if type(scaler) == StandardScaler:
            # much more performant than using inverse_transform
            return x_scaled * scaler.scale_ + scaler.mean_
        else:
            return scaler.inverse_transform([[x_scaled]])[0][0]
            
    def case_to_readable(self, case : tuple, prune_eos = False):
        result = []
        for i in range(case[0][0].shape[1]):
            if case[0][self.concept_name_id][0, i]:
                if prune_eos and case[0][self.concept_name_id][0,i] == self.eos_id:
                    continue
                event = self.event_to_readable(case, i)
                result.append(event)
        return result

    def event_to_readable(self, case : tuple, i : int):
        result = {}
        # decode categorical attributes        
        for j in range(len(case[0])):
            # attribute_name = self.dataset.all_categories[0][j][0]
            attribute_name = self.prefix_cat_attributes[j]
            value = case[0][j][0, i].item()
            result[attribute_name] = self.inverted_prefix_categories[j][value] if value else None
        
        # decode numerical attributes
        for j in range(len(case[1])):
            attribute_name = self.prefix_num_attributes[j]
            
            attribute_value = case[1][j][0, i].item()
            result[attribute_name] = \
                self.dataset.encoder_decoder.continuous_encoders[attribute_name].inverse_transform([[attribute_value]]).item()
        return result