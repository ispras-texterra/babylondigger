from typing import List

import babylondigger.neural.tagger.neural_tagger as neural_tagger
import babylondigger.tagger as tagger

__trainer_factories = {
    'stub': tagger.StubTaggerTrainer,
    'neural': neural_tagger.NeuralTaggerTrainer,
    'neural_pretrained': neural_tagger.PretrainedTaggerTrainer
}


def get_trainer(name: str) -> tagger.TaggerTrainerInterface:
    if name not in __trainer_factories:
        raise NotImplementedError("Unknown method")
    return __trainer_factories[name]()


def trainer_names() -> List[str]:
    return list(__trainer_factories.keys())
