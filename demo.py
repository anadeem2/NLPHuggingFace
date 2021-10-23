from logging import NOTSET
from absl import app, flags, logging
from pytorch_lightning import trainer
import torch as th
import pytorch_lightning as pl
import nlp
from torch.optim import optimizer
import transformers
import sh

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seqLen', 32, '')
flags.DEFINE_integer('bs', 32, '')
flags.DEFINE_integer('percent', 20, '')


FLAGS = flags.FLAGS

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model) 
        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(x['text'], max_length=FLAGS.seqLen, pad_to_max_length=True)['input_ids']
            return x

        def _prepareDS(type):
            ds = nlp.load_dataset('imdb', split= f'{type}[:{FLAGS.percent}%]')
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.trainDS, self.testDS = map(_prepareDS, ('train', 'test'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        preds = self.model(input_ids, mask)
        return preds.logits
    
    def training_step(self, batch, idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss':loss, 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss':loss, 'val_acc': acc}
        return {**out, 'log': out}
    
    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.trainDS,
            batch_size = FLAGS.bs,
            drop_last=True,
            shuffle= True,
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.testDS,
            batch_size = FLAGS.bs,
            drop_last=False,
            shuffle= False,
        )
#Change SGD to Adam, add weightDecay, weight_decay= FLAGS.weightDecay
    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr = FLAGS.lr,
            momentum= FLAGS.momentum
        )


def main(_):
    model = IMDBSentimentClassifier()
    trainer = pl.Trainer(
        default_root_dir = 'logs',
        gpus= (1 if th.cuda.is_available() else 0),
        max_epochs = FLAGS.epochs,
        logger= pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0)
    )
    trainer.fit(model)
    



if __name__ == '__main__':
    app.run(main)


"""
Questions:
#on huggingface there was a transformers.PreTrainedTokenizerFast - what are those used for? Why does Bert have a speicial tokenizer, I thought embeddings were stardardized? Also what is going on w/ fast vs regular?

Modifications:
#Change pad to max to false, fastAI has a function where is puts similar size sentences togeather to avoid padding too much

Notes:
Make jupyter notebook block in middle of code: import IPython ; IPython.embed() ; exit(1)
"""