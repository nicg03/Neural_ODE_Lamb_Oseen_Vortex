"""
File di configurazione contenente tutti i parametri costanti del progetto.
"""

# Parametri di training
TRAINING_CONFIG = {
    'num_epochs': 20000,
    'learning_rate': 1e-2,
    'scheduler_patience': 100,
    'scheduler_factor': 0.5,
    'print_every': 50
}

# Parametri del dataset
DATASET_CONFIG = {
    'grid_resolution': 200,
    'domain_size': 2.0,
    'time': 1.0,
    'viscosity': 0.01
}

# Parametri dei vortici
VORTEX_CONFIG = {
    'vortex1': {
        'gamma': 1.0,
        'x': -0.5,
        'y': 0.0
    },
    'vortex2': {
        'gamma': -0.8,
        'x': 0.5,
        'y': 0.5
    },
    'vortex3': {
        'gamma': 0.6,
        'x': 0.0,
        'y': -0.5
    }
}

# Parametri della rete neurale
MODEL_CONFIG = {
    'input_size': 3,
    'hidden_size': 128,
    'output_size': 2,
    'num_hidden_layers': 3
}

# Parametri per la valutazione
EVAL_CONFIG = {
    'trajectory': {
        'initial_point': [0.0, 0.0],
        'time_steps': 100,
        'time_range': [0, 2]
    }
}

# Percorsi dei file
PATHS = {
    'dataset': 'data/lamb_oseen_dataset.pt',
    'model': 'models/lamb_oseen_multivortex_model.pth',
    'metrics': 'metrics/training_metrics.npz',
    'plots': {
        'training_loss': 'img/training_loss.pdf',
        'velocity_fields': 'img/velocity_fields.pdf',
        'trajectory': 'img/trajectory.pdf'
    }
} 