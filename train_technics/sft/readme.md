# SFT memo

## SFT de base 

Le dataloader `DataCollatorForLanguageModeling`

Input: Prompt d'entrée
Labels: Prompt d'entrée

SFT trainer et juste une class qui utilise les models du type `AutoModelForCausalLM`

## SFT Train

L'idée et que le model predise ce que doit dire le bot

En input le prompt du user, en output ce que doit dire le bot

Loss: CrossEntropyLoss

1- on format la conversation en entier

````
<|prompter|>I don\'t understand the riddle about 3 chests, can you explain it to me?<|endoftext|><|assistant|>I suppose you mean this one:\n\n>There are three chests: <|endoftext|><|prompter|>That was the riddle, thanks for explaining!<|endoftext|><|assistant|>You\'re welcome. Anything else I can assist you with?<|endoftext|><|prompter|>Sure. Make a difficult riddle and let me try to guess the answer!<|endoftext|><|assistant|>Sure! Here\'s a classic riddle with an unexpected twist:\n\nRiddle<|endoftext|>
````

2- on la tokenize

````
{'input_ids': [50279, 42, 1053, 626, 2096, 253, 391, 3209, 670, 495, 1161, ...], "attention_mask": [1, 1, 1, ...]}
````

3- on mask la réponse du bot (labels) dans les tokens, pour ne pas donner la réponse

On garde les input_ids inchanger

````
# for each token an integer indicating the index of the message it belongs to. Just to create the label mask.
# Label mask is true when predicting a token that is part of the answer, false otherwise.
# TEXT:             Question: Hello, how are you? Answer: I am fine. Question: What is your name? Answer: My name is John.
# MESSAGE_INDICES:  0         0      0   0   0    1       1 1  1     2         2    2  2    2     3       3  3    3  3
# LABEL_MASK:       0         0      0   0   0    1       1 1  1     0         0    0  0    0     1       1  1    1  1
````

4- on genarae le format global pour le model

**input_ids**: Input pour le model
**targets**: idem que input_ids mais on decal les tokens de 1 vers la droite
**label_masks**: Mask pour savoir ou sont les réponses du bot dans targets

5- Le model dois avoir une linear layer

````
embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
````

Pour chaque token example une phrase de 400 token, on vas les passer dans la linear (si vocab size = 1000)
sela donne [batch, 400, 1000]

6- On applique une **CrossEntropyLoss** mais seulment sur les zone ou label_masks et True (réponse du bot)
En retire toutes les token qui ne sont pas sux du bot

On fait un simple torch.argmax(input, dim=1).shape de ce qui a été prédit par le model comme de la classification

````python
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super().__init__(weight, size_average, ignore_index, reduce, "none")
        self._reduction = reduction

    def forward(self, input, target, mask=None):
        input = input.view(-1, input.size(-1))
        target = target.view(-1)

        if mask is not None:
            mask = mask.view(-1).bool()
            input = input[mask]
            target = target[mask]

        size = target.numel()

        loss = super().forward(input, target)

        if self._reduction == "none":
            return loss
        return loss.sum() / (size + 1e-8)
````

