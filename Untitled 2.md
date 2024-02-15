```
Map: 100%|█████████████████████████████████████████| 210289/210289 [00:06<00:00, 30064.71 examples/s]                                                                                                               
  0%|                                                                         | 0/60 [00:00<?, ?it/s]RuntimeError: Triton Error [HIP]:  Code: 1, Messsage: invalid argument                                         
                                                     
The above exception was the direct cause of the following exception:                                                                                                                                                
                                                                                                          
Traceback (most recent call last):                   
  File "/home/lhl/bnb/test-unsloth.py", line 60, in <module>                                                                                                                                                        
    trainer.train()                                  
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/trl/trainer/sft_trainer.py", line 323, in train                                                                                                  
    output = super().train(*args, **kwargs)          
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                               
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/transformers/trainer.py", line 1539, in train                                                                                                    
    return inner_training_loop(                                                                           
           ^^^^^^^^^^^^^^^^^^^^                      
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/transformers/trainer.py", line 1869, in _inner_training_loop                                                                                     
    tr_loss_step = self.training_step(model, inputs)                                                      
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                      
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/transformers/trainer.py", line 2772, in training_step                                                                                            
    loss = self.compute_loss(model, inputs)          
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^          
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/transformers/trainer.py", line 2795, in compute_loss                                                                                             
    outputs = model(**inputs)                        
              ^^^^^^^^^^^^^^^                        
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1529, in _wrapped_call_impl                                                                                    
    return self._call_impl(*args, **kwargs)          
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^          
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1538, in _call_impl                                                                                            
    return forward_call(*args, **kwargs)             
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^             
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/accelerate/utils/operations.py", line 817, in forward                                                                                            
    return model_forward(*args, **kwargs)            
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^            
  File "/home/lhl/miniforge3/envs/bnb/lib/python3.11/site-packages/accelerate/utils/operations.py", line 805, in __call__                                                                                           
    return convert_to_fp32(self.model_forward(*args, **kwargs))                                           
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                            

```