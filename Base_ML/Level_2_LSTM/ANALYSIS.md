#Custom LSTM Implementation

#The role of each gate 

Since built-in modules like nn.LSTM were strictly banned, I had to manually construct the cell using basic linear transformations. The input gate basically decides what new information is worth writing to the cell state, while the forget gate handles wiping out historical data that is no longer useful. A critical detail here: I explicitly initialized the forget gate's bias to 1.0 (nn.init.constant_(self.wf.bias, 1.0)). If you don't do this, the network starts out by forgetting everything, and the gradients die immediately. The candidate state proposes the new values to add, and the output gate acts as a filter to decide what part of the internal state actually gets exposed to the next layer. 

#How information is retained or forgotten

The entire memory mechanism runs on the cell state update equation: ct = ft * c_prev + it * gt. It is a balancing act. If the forget gate (f_t) fires close to 1, it carries the historical memory (c_prev) forward. If it drops to 0, that history is erased. New data only gets committed to the cell state if both the input gate and the candidate state agree that the current timestep's features are important. 

#Training stability

Training a custom recurrent network from scratch is incredibly unstable. Unrolling the math over a long sequence causes the weights to multiply repeatedly, which inevitably leads to exploding gradients. During my initial runs, the loss would just blow up to NaN. To fix this, I had to hardcode gradient clipping (nn.utils.clip_grad_norm_) with a max_norm of 1.0 into the training loop. This prevents the math from tearing itself apart and keeps the Adam optimizer stable. 

#Sequence length considerations 

I set the input sequence to 72 hours (exactly 3 days). I found this to be the sweet spot. Weather patterns rely heavily on diurnal (daily) cycles, so a 3-day look-back gives the model enough context to recognize daily temperature peaks and troughs without unrolling the computational graph so far back that optimization becomes impossible due to degrading memory. 
Forecasting challenges Weather forecasting is inherently chaotic, and predicting 12 hours into the future forces the network to differentiate between macro-weather events (like a sudden cold front) and standard micro-fluctuations (like it just getting darker and colder at night). 

#Explanation of design choices 

The most frustrating requirement was the evaluation metrics. The prompt mandated classification metrics (Precision, Recall, F1 Score, Confusion Matrix) for what is strictly a continuous MSE regression task. Instead of breaking the continuous nature of the network, I engineered a workaround. I wrote a categorize_temp pipeline to bucket the model's continuous temperature predictions into four discrete classes ('Very Cold', 'Cold', 'Mild', 'Warm'). This allowed me to generate a perfectly valid classification report and confusion matrix without ruining the actual regression training. I also maxed out the architecture at 2 layers with a 128 hidden dimension to stay safely within the 32-256 limit, and added a 0.2 dropout rate to stop it from memorizing the sequence. 
