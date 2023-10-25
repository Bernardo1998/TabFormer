from torch.nn import CrossEntropyLoss
import torch
from torch.distributions import Categorical

import pandas as pd

from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutput


from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel
from transformers.utils import logging
from pathlib import Path
from tqdm import tqdm

DEBUG = False

class TabFormerGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, vocab):
        super().__init__(config)
        self.vocab = vocab

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=True,    
            return_dict=None
    ):        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        # lm_logits : [bsz x seq_len x vsz]
        # labels    : [bsz x seq_len]
        # When flatten is set to True:
        # seq_len = num_transactions * (num_columns + 2)  --> plus 2 because each transaction has BOS and EOS padding

        #for field_name in ['Quantity', 'UnitPrice', 'Country', 'InvoiceDate']:
        #    print(self.vocab.token2id[field_name])

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_labels = labels[..., 1:-1].contiguous()  # Remove first and last label: [BOS] and [EOS] tokens
            #print("lm_logits:", lm_logits.shape)
            shift_logits = lm_logits[..., :-2, :].contiguous()  # Line up logits accordingly
            #print("shift_logits:", shift_logits.shape)

            seq_len = shift_logits.size(1)
            total_lm_loss = 0
            field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=True)

            for field_idx, field_name in enumerate(field_names):
                #print("field_name",field_name)
                col_ids = list(range(field_idx, seq_len, len(field_names)))
                global_ids_field = self.vocab.get_field_ids(field_name)
                lm_logits_field = shift_logits[:, col_ids, :][:, :, global_ids_field]  # bsz * 10 * K
                lm_labels_field = shift_labels[:, col_ids]
                lm_labels_local_field = self.vocab.get_from_global_ids(global_ids=lm_labels_field,
                                                                       what_to_get='local_ids')

                loss_fct = CrossEntropyLoss()
                if DEBUG:
                    print(f'Min label value: {torch.min(lm_labels_local_field)}')
                    print(f'Max label value: {torch.max(lm_labels_local_field)}')
                    print(f'Logits dimensions: {lm_logits_field.size()}')
                    reshaped_logits = lm_logits_field.view(-1, len(global_ids_field))
                    reshaped_labels = lm_labels_local_field.view(-1)
                    print(f'Reshaped logits dimensions: {reshaped_logits.size()}')
                    print(f'Reshaped labels dimensions: {reshaped_labels.size()}')
                    print(f'Are there any negative values in logits? {torch.any(reshaped_logits < 0)}')
                    print(f'Are there any negative values in labels? {torch.any(reshaped_labels < 0)}')
                    print(f'Are there any NaN values in logits? {torch.isnan(reshaped_logits).any()}')
                    print(f'Are there any NaN values in labels? {torch.isnan(reshaped_labels).any()}')


                lm_loss_field = loss_fct(lm_logits_field.view(-1, len(global_ids_field)),
                                         lm_labels_local_field.view(-1))
                total_lm_loss += lm_loss_field

            outputs = (total_lm_loss,) + outputs

        #return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
        
        return CausalLMOutput(
            loss=total_lm_loss if labels is not None else None,
            logits=lm_logits,
            #past_key_values=transformer_outputs[1] if len(transformer_outputs) > 1 else None,
            hidden_states=transformer_outputs[2] if len(transformer_outputs) > 2 else None,
            attentions=transformer_outputs[3] if len(transformer_outputs) > 3 else None
        )
    
    def autoregressive_generate(self, input_ids, max_length=100, temperature=1.0,condition=None):
        """
        Args:
        - input_ids (torch.Tensor): a tensor of shape (batch_size, seq_len) containing the input IDs.
        - max_length (int): the maximum sequence length to generate.
        - temperature (float): the temperature for sampling. Higher values make the sampling more random.
        - condition: (dictionary): global ids of conditional context columns. Note that we can set a condition as None, then the condition will be set by the first generation. 

        Returns:
        - generated_ids (torch.Tensor): a tensor of shape (batch_size, generated_seq_len) containing the generated IDs.
        """
        self.eval()  # Set the model to evaluation mode
        field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=True)
        idx_current_field = 0
        with torch.no_grad():  # Disable gradient computation
            generated_ids = input_ids
            batch_size = input_ids.size(0)
            for _ in range(max_length - input_ids.size(1)):
                current_field = field_names[idx_current_field]
                if condition is not None and current_field in condition and condition[current_field] is not None: 
                    # Use the given condition
                    next_token = condition[current_field]
                else:
                    # Get the model's output if no condition given
                    outputs = self(input_ids=generated_ids)
                    logits = outputs.logits[:, -1, :] / temperature  # Get the logits of the last token and apply temperature
                    # Decide the next field to be sampled and select its logits
                    idx_current_field = idx_current_field + 1 if idx_current_field < (len(field_names)-1) else 0
                    global_ids_field = self.vocab.get_field_ids(current_field)
                    lm_logits_field = logits[-1, global_ids_field]
                    # Sample the next id based on probs of this field
                    probs = torch.nn.functional.softmax(lm_logits_field, dim=-1)
                    m = Categorical(probs)
                    predicted_local_id = m.sample().unsqueeze(-1)
                    next_token = self.vocab.get_from_local_ids(predicted_local_id,current_field)
                    if condition is not None and current_field in condition and condition[current_field] is None: 
                        condition[current_field] = next_token
                # Check for the EOS token
                if (next_token == 6):
                    break
                next_token_tensor = torch.tensor([[next_token]]).to(generated_ids.device)
                # Append the next token to the generated sequence
                generated_ids = torch.cat((generated_ids, next_token_tensor), dim=1)
        return generated_ids
       
    def generate_table(self, n, condition=None, max_length=1024,prompt=None):
        '''
            condition: a dictionary in form: {'field_name':local_id}
        '''
        # Assuming you have the sentence start ID        
        prompt = torch.tensor([[5]]) if prompt is None else prompt
        print("Condition:", condition)

        # Generate output from the model using the sentence start ID as prompt
        decoded_sequences = []
        columns = self.vocab.get_field_keys() # Get original data columns
        print(columns)
        #max_length = n * len(columns) + 2 # + 2 for BOS/EOS
        # Using a beam search for better quality text, but you can modify this as required
        for i in tqdm(range(n)):
            #try:
            prompt_i = prompt if len(prompt) == 1 else prompt[i]
            generated_output = self.autoregressive_generate(prompt_i.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), max_length=max_length, condition=condition)

            # Decode the generated output using the vocabulary object's method
            decoded_output = self.vocab.get_from_global_ids(generated_output[0], 'tokens')
            print(decoded_output)

            # Convert to string
            generated_sequence = self.convert_to_dataframe(decoded_output, columns)
            #generated_sequence.insert(loc=0, column='User', value=i)
            if generated_sequence is not None:
                decoded_sequences.append(generated_sequence)
                print(generated_sequence)
            #except Exception as e:
            #    print(f"Error {e} generating sequence {i}. Skipping.")

        return pd.concat(decoded_sequences)

    def convert_to_dataframe(self, pairs, columns):

        # Initialize an empty list to collect rows
        rows = []

        # Initialize an empty dictionary to collect column-value pairs for the current row
        current_row = {}

        # Traverse the pairs
        for pair in pairs:
            # Split the pair into column and value
            column, value = pair.split('_')
            # Don't convert special tokens
            if column == 'SPECIAL':
                continue

            # If this column is in the desired columns list, add it to the current row
            if column in columns:
                # If this column already exists in the current row, it means we've completed a row
                if column in current_row:
                    # Add the completed row to the rows list
                    rows.append(current_row)
                    # Start a new row
                    current_row = {}

                # Add this column-value pair to the current row
                current_row[column] = value

        # Don't forget to add the last row if it's not empty
        if current_row:
            rows.append(current_row)
            print("current_row",current_row)

        # Convert the list of rows to a DataFrame
        df = pd.DataFrame(rows)

        return df

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Ensure the configuration is correct
        config = kwargs.pop("config", None)
        if config is None:
            config = GPT2Config.from_pretrained(pretrained_model_name_or_path)

        # Create the model
        model = cls(config, *model_args, **kwargs)
        
        # Load the state dictionary
        state_dict = None
        try:
            state_dict = PreTrainedModel._state_dict_from_pretrained(
                pretrained_model_name_or_path,
                proxy=kwargs.pop("proxy", None),
            )
        except UserWarning as err:
            pass

        if state_dict is None:
            return model

        # Update the state dictionary to match the class
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

        return model