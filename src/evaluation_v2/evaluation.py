import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 

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
                 decoder_cat = None,
                 decoder_num = None):
        
        self.model = model
        self.dataset = dataset

        self.concept_name = concept_name
        # Index of activity categories in all categories
        self.concept_name_id = [i for i, cat in enumerate(self.dataset.all_categories[0]) if cat[0] == self.concept_name][0]
        # Index of EOS token in activity categories
        self.eos_id = [v for k, v in self.dataset.all_categories[0][self.concept_name_id][2].items()if k == eos_value][0]
        
        self.growing_num_values = growing_num_values

        # Event attributes used by encoder as input
        self.prefix_cat_attributes = self.model.enc_feat[0]
        self.prefix_num_attributes = self.model.enc_feat[1]
        prefix_categories = [cat_tuple for cat_tuple in self.dataset.all_categories[0] if cat_tuple[0] in self.prefix_cat_attributes]
        self.inverted_prefix_categories = [{v: k for k, v in s[2].items()} for s in prefix_categories]
        
        # Event attributes for decoder as input and output
        self.all_cat_attributes = decoder_cat if decoder_cat else [cat[0] for cat in self.dataset.all_categories[0]]
        self.all_num_attributes = decoder_num if decoder_num else [num[0] for num in self.dataset.all_categories[1]]
        suffix_categories = [ cat_tuple for cat_tuple in self.dataset.all_categories[0] if cat_tuple[0] in self.all_cat_attributes]
        self.inverted_suffix_categories = [{v: k for k, v in s[2].items()} for s in suffix_categories]
        
        # Get all cases to be iterated
        self.cases = self._get_cases_from_dataset()

    def _get_cases_from_dataset(self):
        """
        Change: for version 2: 
        
        idea:
        - filter for case ids and take only those that are stored in a list.
        - store the tensors and eos padds with the same indices in a separated list
        - store filter all  
        
        train_set[0][0] = 'Case 10' -> event is case id
        
        """
        
        """
        cases = {}
        for event in self.dataset:
            # iterate
            suffix = event[0][self.concept_name_id][-self.dataset.encoder_decoder.min_suffix_size:]
            if torch.all(suffix  == self.eos_id).item():
                cases[event[2]] = event
        return cases
        """
        # Add always the first occurence of the case id to the cases
        cases = {}
        for padded_case in self.dataset: 
            case_id = padded_case[0]
            if case_id not in cases:
                cases[case_id] = padded_case 
        return cases
        
        
    def _iterate_case(self, case: tuple) -> Iterator[tuple]:
        """Yield prefix/suffix tuples for a single encoded case.

        Expected case layout (from EventLogDataset.__getitem__):
            0 -> case identifier (ignored here)
            1 -> tuple of categorical tensors (window_size,)
            2 -> tuple of numerical tensors (window_size,)
            3 -> EOS mask (window_size,) used to keep a single EOS token
            4 -> zero-padding mask (unused here but passed along in the tuple)
            5 -> static categorical tensor
            6 -> static numerical tensor
            7 -> optional Petri net markings/metadata

        Older structures with only (categorical_tensors, numerical_tensors) are also supported.
        """

        _, categorical_tensors, numerical_tensors, eos_mask, zero_mask, static_cats, static_nums, *_ = case
        static_atts = self._prepare_static_inputs(static_cats, static_nums)

        cat_attrs = [t.clone() for t in categorical_tensors]
        num_attrs = [t.clone() for t in numerical_tensors]

        # total padded case length
        window_size = cat_attrs[0].shape[0]
        # zero_mask: 1 for real events (a,b,c,d), 0 for left padding
        if zero_mask is not None:
            zm = zero_mask.squeeze().bool()
        else:
            zm = torch.ones(window_size, dtype=torch.bool)

        # Mask away left-padding selects only real events: 
        masked_cat = [t[zm].clone() for t in cat_attrs]
        masked_num = [t[zm].clone() for t in num_attrs]

        # Reduce case by eos mask:
        if eos_mask is not None:
            eosm = eos_mask.squeeze().bool()
        else:
            eosm = torch.ones(window_size, dtype=torch.bool)
        eos_zero_red = eosm[zm].clone()
            
        eos_red_masked_cat = [t[eos_zero_red].clone() for t in masked_cat]
        split_len = eos_red_masked_cat[0].shape[0]-2
              
        # add EOS if not present and pad other attributes with their last value
        event_label_tensor = masked_cat[self.concept_name_id]
        if torch.all(event_label_tensor != self.eos_id):
            for idx, tensor in enumerate(masked_cat):
                if tensor.numel() == 0:
                    continue
                if idx == self.concept_name_id:
                    eos_tensor = tensor.new_tensor([self.eos_id])
                    masked_cat[idx] = torch.cat([tensor, eos_tensor], dim=0)
                else:
                    masked_cat[idx] = torch.cat([tensor, tensor[-1:].clone()], dim=0)
            for idx, tensor in enumerate(masked_num):
                if tensor.numel() == 0:
                    continue
                masked_num[idx] = torch.cat([tensor, tensor[-1:].clone()], dim=0)
        
            # If no eos token in the case already then: add EOS but reduce from new length:       
            split_len = masked_cat[0].shape[0]-2

        # create dummy prefix and suffix pairs
        current_prefix = ([torch.zeros_like(cat_attribute).unsqueeze(0) for cat_attribute in masked_cat], [torch.zeros_like(num_attribute).unsqueeze(0) for num_attribute in masked_num])
        current_suffix = ([torch.clone(cat_attribute).unsqueeze(0) for cat_attribute in masked_cat], [torch.clone(num_attribute).unsqueeze(0) for num_attribute in masked_num])

        prefix_len = 0
        n_cat_attrs = len(current_prefix[0])
        n_num_attrs = len(current_prefix[1])
        # How many prefix, suffix pairs should be build -> create (case len-1) prefix, suffix pairs (still contains one EOS?)
        for i in range(split_len):
            # Build iterative the prefix, suffixes:
            for j in range(n_cat_attrs):
                current_prefix[0][j][0] = torch.roll(current_prefix[0][j][0], shifts=-1)
                current_prefix[0][j][0, -1] = masked_cat[j][i]
                
                current_suffix[0][j][0] = torch.roll(current_suffix[0][j][0], shifts=-1)
                current_suffix[0][j][0, -1] = 0

            for j in range(n_num_attrs):
                current_prefix[1][j][0] = torch.roll(current_prefix[1][j][0], shifts=-1)
                current_prefix[1][j][0, -1] = masked_num[j][i]
                
                current_suffix[1][j][0] = torch.roll(current_suffix[1][j][0], shifts=-1)
                current_suffix[1][j][0, -1] = 0

            prefix_len += 1

            prefix = ([tensor.clone() for tensor in current_prefix[0]], [tensor.clone() for tensor in current_prefix[1]])
            suffix = ([tensor.clone() for tensor in current_suffix[0]], [tensor.clone() for tensor in current_suffix[1]])
            
            # fix both
            yield prefix_len, prefix, static_atts, suffix
            
    def _prepare_static_inputs(self, cats_static, nums_static):
        def _ensure_batch(tensor):
            if tensor is None or tensor.numel() == 0:
                return None
            return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

        static_cat = _ensure_batch(cats_static)
        static_num = _ensure_batch(nums_static)

        if static_cat is None and static_num is None:
            return None
        return (static_cat, static_num)    
                
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

    def _predict_suffix_with_means(self, prefix_len, prefix, statics):
        # disable dropout
        dropout_rates = self._disable_model_dropout(self.model)
        self.model.decoder 
        
        # Prediction by model
        prediction, (h, c), z = self.model.inference(prefix=prefix, static_inputs=statics)
        
        suffix = []
        max_iteration = self.dataset.encoder_decoder.window_size - self.dataset.encoder_decoder.min_suffix_size - prefix_len
        i = 0
        eos_predicted = lambda prediction : torch.argmax(prediction[0][0][self.concept_name+'_mean']) == self.eos_id
        
        last_means = {a+'_mean' : prefix[1][self.all_num_attributes.index(a)][:,-1].unsqueeze(1) for a in self.growing_num_values}
        
        while i <= max_iteration and not eos_predicted(prediction):
            cat_prediction = {k : torch.argmax(cat_pred, keepdim=True) for k, cat_pred in prediction[0][0].items()}
                        
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