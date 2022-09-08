from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_transformer')

dataset_params = {
    'name':'hw',
    'data_root':'../dataset/pres/vietocr',
    'train_annotation':'train.txt',
    'valid_annotation':'val.txt'
}

params = {
         'print_every':200,
         'valid_every':15*200,
          'iters':20000,
          'export':'./weights/transformerocr.pth',
          'metrics': 2000
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'
trainer = Trainer(config, pretrained=True)
trainer.config.save('config.yml')
trainer.train()