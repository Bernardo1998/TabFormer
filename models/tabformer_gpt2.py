from torch.nn import CrossEntropyLoss
import torch

import pandas as pd

from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutput

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
    
    def generate_table(self, n, vocab, prompt=None):
        '''
            prompt: either one prompt for all sequences, or separate prompt for each sequence.
        '''
        # Assuming you have the sentence start ID
        if prompt is None:
            prompt = torch.tensor([[5,8,13,24,34,38,374,638.700,1000,1078,1084]])  # replace with your actual sentence start ID
        elif torch.is_tensor(prompt):
            if len(prompt) != 1 and len(prompt) != n:
                raise ValueError(f"Incompatible prompt length! Expected 1 or {n} but got {len(prompt)}")
        else:
           raise ValueError(f"Incompatible prompt type: {type(prompt)}. Expected torch.Tensor.")

        # Generate output from the model using the sentence start ID as prompt
        decoded_sequences = []
        columns = vocab.get_field_keys() # Get original data columns
        print(columns)
        # Using a beam search for better quality text, but you can modify this as required
        for i in range(n):
            prompt_i = prompt if len(prompt) == 1 else prompt[i]
            generated_output = self.generate(prompt_i.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), max_length=100, num_beams=5)

            # Decode the generated output using the vocabulary object's method
            decoded_output = vocab.get_from_global_ids(generated_output[0], 'tokens')

            # Convert to string
            generated_sequence = self.convert_to_dataframe(decoded_output, columns)
            generated_sequence.insert(loc=0, column='User', value=i)
            if generated_sequence is not None:
                decoded_sequences.append(generated_sequence)
                print(generated_sequence)

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

