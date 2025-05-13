from transformers import T5Tokenizer, T5EncoderModel
import torch

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")


# Save model to defined path for future use no need to download again aprx 5gb 

tokenizer.save_pretrained("/home/onur/Desktop/Project/model/prot_t5_xl_half_uniref50-enc")
model.save_pretrained("/home/onur/Desktop/Project/model/prot_t5_xl_half_uniref50-enc")





