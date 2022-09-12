
import torch
import pandas as pd
from transformers.models.auto import AutoConfig, AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

class TransformerONET:
    def __init__(self, model_name: str, tokenizer = None):
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
    
    def force_prediction(self, predicted_probs_df):
        import pandas
        import numpy as np
        
        predicted_probs_df[predicted_probs_df > 0.5] = 1
        predicted_prob_df_nomax = predicted_probs_df[predicted_probs_df.max(axis=1) < 0.5]
        predicted_prob_df_nomax.values[range(len(predicted_prob_df_nomax.index)), 
                                          np.argmax(predicted_prob_df_nomax.values, axis=1)] = 1
        all_predicted = predicted_prob_df_nomax.combine_first(predicted_probs_df)
        all_predicted[all_predicted != 1] = 0
        return all_predicted
        
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        import torch
        import pandas as pd
        import numpy as np
        from datasets import Dataset
        
        with torch.no_grad():
            # Tokenize the passed data set
            data = Dataset.from_pandas(data)
            inputs = self.tokenizer(data['Task'], padding=True, return_tensors='pt')
        
            # Send data set to GPU for predictions by GWA column 
            if self.model.device.index != None:
                torch.cuda.empty_cache()
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(self.model.device.index)    

            # Generate model predictions and retrieve the produced probabilities
            predictions = self.model(**inputs)
            probs = torch.nn.Softmax(dim=1)(predictions.logits)
            probs = probs.detach().cpu().numpy()
            
            # Force at least one prediction for each task & convert numeric column codes to original GWA code labels
            outputs = self.force_prediction(pd.DataFrame(probs[:,]))      
            outputs = outputs.rename(self.config.id2label, axis=1)
       
        return outputs
        
def _load_pyfunc(path):
    import os
    return TransformerONET(os.path.abspath(path))
