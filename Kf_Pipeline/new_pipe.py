import kfp
from kfp import dsl

def calculate_rul():
    vop = dsl.PipelineVolume(pvc="datavol")
    return dsl.ContainerOp(
        name='Preprocess data to calculate RUL and remove useless columns',
        image='sachin944/nasa_pa:step1_rulv4',
        arguments=[],
        file_outputs={
            'train_df': '/app/train_df.csv',
            'fd_001_test': '/app/fd_001_test.csv',
            'unit_number': '/app/unit_number.csv',
            'RUL': '/app/RUL.csv'
        },
        pvolumes={"/mnt": vop}
    )#.container.set_image_pull_policy('Always')

def train_test_preprocess(train_df, fd_001_test):
    return dsl.ContainerOp(
        name='Preprocess train test data',
        image='sachin944/nasa_pa:step2_rf_xgbv2',
        arguments=[
            '--train_df', train_df,
            '--fd_001_test', fd_001_test
        ],
        file_outputs={
            'X': '/app/X.npy',
            'Y': '/app/Y.npy',
            'X_001_test': '/app/X_001_test.npy'
        }
    )#.container.set_image_pull_policy('Always')

def train_test_preprocess_lstm(train_df, fd_001_test, unit_number, RUL):
    return dsl.ContainerOp(
        name='Preprocess train test data and generate sequence for LSTM',
        image='sachin944/nasa_pa:step2_lstmv1',
        arguments=[
            '--train_df', train_df,
            '--fd_001_test', fd_001_test,
            '--unit_number', unit_number,
            '--RUL', RUL
        ],
        file_outputs={
            'seq_array': '/app/seq_array.npy',
            'label_array': '/app/label_array.npy',
            'test_df': '/app/test_df.csv'
        }
    )#.container.set_image_pull_policy('Always')

def train_rf(X, Y):
    return dsl.ContainerOp(
        name='Train Random Forest Regressor',
        image='sachin944/nasa_pa:step3_rfv1',
        arguments=[
            '--X', X,
            '--Y', Y
        ],
        file_outputs={
            'model_rf': '/app/model_rf.pickle'
        }
    )#.container.set_image_pull_policy('Always')

def train_xgb(X, Y):
    return dsl.ContainerOp(
        name='Train XGBOOST Regressor',
        image='sachin944/nasa_pa:step3_xgbv1',
        arguments=[
            '--X', X,
            '--Y', Y
        ],
        file_outputs={
            'model_xgb': '/app/model_xgb.pickle'
        }
    )#.container.set_image_pull_policy('Always')

def train_lstm(seq_array, label_array):
    vop = dsl.PipelineVolume(pvc="datavolmodels")
    return dsl.ContainerOp(
        name='Train LSTM RNN',
        image='sachin944/nasa_pa:step3_lstmv2',
        arguments=[
            '--seq_array', seq_array,
            '--label_array', label_array
        ],
        file_outputs={
            'model_lstm': '/app/last_model.h5'
        },
        pvolumes={"/mnt": vop}
    ).set_gpu_limit(1)#.container.set_image_pull_policy('Always')

def test_rf(model_rf, X_001_test):
    vop = dsl.PipelineVolume(pvc="datavolrf")
    return dsl.ContainerOp(
        name='Test Random Forest Regressor',
        image='sachin944/nasa_pa:step4_rfv1',
        arguments=[
            '--model_rf', model_rf,
            '--X_001_test', X_001_test
        ],
        pvolumes={"/mnt": vop}
    )#.container.set_image_pull_policy('Always')

def test_xgb(model_xgb, X_001_test):
    vop = dsl.PipelineVolume(pvc="datavolxgb")
    return dsl.ContainerOp(
        name='Test XGBOOST Regressor',
        image='sachin944/nasa_pa:step4_xgbv1',
        arguments=[
            '--model_xgb', model_xgb,
            '--X_001_test', X_001_test
        ],
        pvolumes={"/mnt": vop}
    )#.container.set_image_pull_policy('Always')

def test_lstm(model_lstm, test_df):
    return dsl.ContainerOp(
        name='Test LSTM',
        image='sachin944/nasa_pa:step4_lstmv1',
        arguments=[
            '--model_lstm', model_lstm,
            '--test_df', test_df
        ]
    ).set_gpu_limit(1)#.container.set_image_pull_policy('Always')

@dsl.pipeline(
   name='Nasa Turbo Engine Failure Prediction Pipeline',
   description='Builds 3 Regression Models to predict the RUL of Nasa Turbo Engines'
)

def pipeline():
    _calculate_rul = calculate_rul()

    _train_test_preprocess = train_test_preprocess(
        dsl.InputArgumentPath(_calculate_rul.outputs['train_df']),
        dsl.InputArgumentPath(_calculate_rul.outputs['fd_001_test'])
    ).after(_calculate_rul)

    _train_test_preprocess_lstm = train_test_preprocess_lstm(
        dsl.InputArgumentPath(_calculate_rul.outputs['train_df']),
        dsl.InputArgumentPath(_calculate_rul.outputs['fd_001_test']),
        dsl.InputArgumentPath(_calculate_rul.outputs['unit_number']),
        dsl.InputArgumentPath(_calculate_rul.outputs['RUL'])
    ).after(_calculate_rul)

    _train_rf = train_rf(
        dsl.InputArgumentPath(_train_test_preprocess.outputs['X']),
        dsl.InputArgumentPath(_train_test_preprocess.outputs['Y'])
    ).after(_train_test_preprocess)

    _train_xgb = train_xgb(
        dsl.InputArgumentPath(_train_test_preprocess.outputs['X']),
        dsl.InputArgumentPath(_train_test_preprocess.outputs['Y'])
    ).after(_train_test_preprocess)

    _train_lstm = train_lstm(
        dsl.InputArgumentPath(_train_test_preprocess_lstm.outputs['seq_array']),
        dsl.InputArgumentPath(_train_test_preprocess_lstm.outputs['label_array'])
    ).after(_train_test_preprocess_lstm)

    _test_rf = test_rf(
        dsl.InputArgumentPath(_train_rf.outputs['model_rf']),
        dsl.InputArgumentPath(_train_test_preprocess.outputs['X_001_test'])
    ).after(_train_rf, _train_test_preprocess)

    _test_xgb = test_xgb(
        dsl.InputArgumentPath(_train_xgb.outputs['model_xgb']),
        dsl.InputArgumentPath(_train_test_preprocess.outputs['X_001_test'])
    ).after(_train_xgb, _train_test_preprocess)

    _test_lstm = test_lstm(
        dsl.InputArgumentPath(_train_lstm.outputs['model_lstm']),
        dsl.InputArgumentPath(_train_test_preprocess_lstm.outputs['test_df'])
    ).after(_train_lstm, _train_test_preprocess_lstm)

#client = kfp.Client()
#client = kfp.Client(host='pipelines-api.kubeflow.svc.cluster.local:8888')
#client.create_run_from_pipeline_func(ex_pipeline, arguments={})

kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml')

