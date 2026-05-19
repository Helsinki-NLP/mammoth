from huggingface_hub import hf_hub_download 
import importlib.util                                                                                                                      

# Step 1: fetch only mammoth_hub.py from the Hub (< 5 KB, instant)                                                                         
_path = hf_hub_download("chao0524/mammoth-single-artifact-test", "mammoth_hub.py")                                                                               
_spec = importlib.util.spec_from_file_location("mammoth_hub", _path)                                                                       
_mod  = importlib.util.module_from_spec(_spec)                                                                                             
_spec.loader.exec_module(_mod)                                                                                                           
MammothHub = _mod.MammothHub 

# Step 2: load — downloads only eng-spa weights on demand (~1.7 GB)                                                                      
model = MammothHub.from_pretrained("chao0524/mammoth-single-artifact-test", task="bul-eng")                                                                        
inputs = model.src_tokenizer(["Хелзинки е столицата на Финландия."], return_tensors="pt")                                                                              
out    = model.generate(**inputs, num_beams=4, max_new_tokens=128)                                                                         
print(model.tgt_tokenizer.batch_decode(out, skip_special_tokens=True))