from .coco_ee_tagger_del_dataset import COCOEETaggerDELDataset
from .coco_ee_tagger_add_dataset import COCOEETaggerADDDataset
from .coco_ee_inserter_dataset import COCOEEInserterDataset
from .flickr30k_ee_tagger_del_dataset import Flickr30KEETaggerDELDataset
from .flickr30k_ee_tagger_add_dataset import Flickr30KEETaggerADDDataset
from .flickr30k_ee_inserter_dataset import Flickr30KEEInserterDataset


__all__ = [
			"COCOEETaggerDELDataset",
			"COCOEETaggerADDDataset",
			"COCOEEInserterDataset",
			"Flickr30KEETaggerDELDataset",
			"Flickr30KEETaggerADDDataset",
			"Flickr30KEEInserterDataset",
]

DatasetMapTrain = {
			'TASK1': COCOEETaggerDELDataset,
			'TASK2': COCOEETaggerADDDataset,
			'TASK3': COCOEEInserterDataset,
			'TASK4': Flickr30KEETaggerDELDataset,
			'TASK5': Flickr30KEETaggerADDDataset,
			'TASK6': Flickr30KEEInserterDataset,
}

DatasetMapEval = {
			'TASK1': COCOEETaggerDELDataset,
			'TASK2': COCOEETaggerADDDataset,
			'TASK3': COCOEEInserterDataset,
			'TASK4': Flickr30KEETaggerDELDataset,
			'TASK5': Flickr30KEETaggerADDDataset,
			'TASK6': Flickr30KEEInserterDataset,
}
