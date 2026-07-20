#Custom ResNet Implementation

#Brief explanation of architecture choices

For the architecture, I decided to go with a smaller custom ResNet instead of just forcing a massive standard one to work. Since CIFAR-10 images are tiny (32×32), using something huge like ResNet-50 would just overfit almost immediately. I set up 3 stages with channel depths of [64, 128, 256] and 2 residual blocks each. This keeps the model lightweight but deep enough to extract good features. I also threw in some label smoothing (0.1) and weight decay because the network started memorizing the training set a bit too fast during my initial test runs. I used Kaiming initialization for the conv layers to make sure the variance didn't blow up before the ReLU activations.

#Why residual connections help optimization
Honestly, training deep networks from scratch is super unstable because the loss surface gets really messy and non-convex. Residual connections basically force the network to learn the difference (the residual mapping) instead of trying to learn the whole unreferenced mapping from scratch. It makes the optimization path way smoother, so the Adam optimizer doesn't get stuck in local minima or saddle points as easily.

#How skip connections improve gradient flow

Without skip connections, the gradients just kind of die out (vanish) before they reach the earlier layers because they have to pass through way too many activations and weight matrices during backprop. The skip connections act like a direct shortcut. Since the forward pass is basically y=F(x)+x, the derivative naturally has a +1 term in it. That +1 guarantees that at least some gradient flows all the way back cleanly without getting completely diluted by the network's internal weights.

#The effect of network depth on performance

In theory, adding more layers means the model can learn much more complex representations. But in practice, I noticed that for this specific low-res dataset, pushing the depth too much doesn't really help and just balloons the parameter count. With the 3 stages, it learns the shapes and classes perfectly fine. Any deeper, and it just starts overfitting heavily, which is why I had to rely on a pretty aggressive data augmentation pipeline (like Random Erasing and Color Jitter) to keep the validation accuracy climbing as the epochs went on.

#Challenges encountered during implementation

The biggest headache by far was matching the tensor dimensions between the different stages. When the stride goes to 2 to downsample the image, the spatial size halves and the channels double. Because of that, you can't just add the skip connection directly to the output anymore (it throws a shape mismatch error). I had to write a dynamic 1×1 conv projection layer with its own batch norm just to scale the skip connection so the tensor addition would actually work. Also, dealing with dataloader bottlenecks was annoying until I properly configured pin_memory=True and adjusted the workers so the GPU wasn't just sitting there starving for data.
