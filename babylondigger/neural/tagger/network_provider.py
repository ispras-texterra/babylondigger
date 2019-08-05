from babylondigger.neural.network import MultitaskNetworkBuilder

import babylondigger.neural.tagger.shared_builders as shared_builders
import babylondigger.neural.tagger.task_builders as task_builders

_shared_builders = {
    'WE_charCNN': shared_builders.WECharCNNSharedNetworkBuilder,
    'BiLSTM': shared_builders.CharCNNBiLSTMSharedNetworkBuilder,
    'BiLSTM_CNN': shared_builders.CharCNNBiLSTMCNNSharedNetworkBuilder,
    'Dilated_CNN':shared_builders.DilatedCNNSharedNetworkBuilder
    # add graph shared part builders here
}

_task_builders = {
    "pos_dense": task_builders.PosDenseTaskNetworkBuilder,
    "lemma_dense": task_builders.LemmaDenseTaskNetworkBuilder,
    "lemma_suffix": task_builders.LemmaSuffixTaskNetworkBuilder,
    "lemma_combo": task_builders.LemmaCOMBOTaskNetworkBuilder,
    "lemma_binary": task_builders.LemmaSuffixBinaryTaskNetworkBuilder
    # add graph task specific builders here
}


def get_nn_builder(shared, tasks, config, seed):
    shared_builder = _shared_builders[shared['builder']](shared['config'], shared.get('features_config', {}))
    task_builders = [_task_builders[task['builder']](task['config'], task.get('features_config', {})) for task in tasks]
    return MultitaskNetworkBuilder(shared_builder, task_builders, config, seed)
