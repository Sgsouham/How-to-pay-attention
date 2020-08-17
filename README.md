# How-to-pay-attention


## "Attention is all you need" paper implementation.
The Transformer model proposed in the [paper](https://arxiv.org/pdf/1706.03762.pdf) had taken the entire NLP community by6 sti\orm because of it's SOTA peformance in less computation power.

The paper also used a concept known as Attention and showed that only attention mechanism can be used as an interaction between layers to achieve SOTA results. So can we say that ***Any architecture which uses only attention as it's medium of interaction with the corresponding layers can be termed as transformer(maybe a basic one but still)?***  

### The implementation (Simple transformer folder)

The folder shows a simple self attention block with a layer norm over the sum of residual connection of the input and the self attention and how this small part can be used for classifying text or generating new text.

- Classify text

    Doesnt contain any masking.
- Generate Text

    Contains masking of the texts to predict the next word.
    
Though the actual architecture is much more complex.

![image](https://miro.medium.com/max/1252/1*JuGZaZcRtmrtCEPY8qfsUw.png)

### The main Components

    Q) What are this big blocks with Nx written beside them?
    A) The left block denotes the Encoder and the right part denotes the Decoder.
    
    Q) What are the component of these blocks?
    A) The concept of MultiHead Attention is the main. It contains the scaled dot product of self-attention to generalize the "attentive portions over a varied          (query, key, value) set. 
    
    
    


